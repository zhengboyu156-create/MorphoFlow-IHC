"""Match the local MAPS-GAN MIST-HER2 comparison metrics for Flow-40 weights."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from morphoflow_ihc.config import read_config
from morphoflow_ihc.engine import precision
from morphoflow_ihc.model import StainFlow
from morphoflow_ihc.trajectory import sample_rgb


DEFAULT_FIJI = Path(r"D:\Fiji\Fiji\fiji-windows-x64.exe")
DEFAULT_MACRO = Path(__file__).resolve().with_name("direct_fiji_intden.ijm")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_map(folder):
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            if path.stem in result:
                raise RuntimeError(f"duplicate image stem: {path.stem} in {folder}")
            result[path.stem] = path
    return result


def run_fiji(tag, sources, root, executable, macro_source):
    # Keep all paths passed to Fiji ASCII-only. The user-facing result folder is
    # on a Chinese-named Windows path, which can leave the Fiji launcher idle.
    stage = Path(__file__).resolve().parent / "fiji_ascii" / tag
    input_dir = stage / "images"
    input_dir.mkdir(parents=True, exist_ok=True)
    for stem, src in sources.items():
        dst = input_dir / (stem + src.suffix.lower())
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    output_csv = stage / "intden.csv"
    output_csv.unlink(missing_ok=True)
    macro = stage / "batch.ijm"
    shutil.copy2(macro_source, macro)
    log = stage / "fiji.log"
    command = [str(executable), "--jaunch-skip-console-check", "--allow-multiple",
               "-macro", str(macro), f"{input_dir},{output_csv}"]
    with log.open("w", encoding="utf-8", errors="replace") as stream:
        proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=3 * 60 * 60)
    if not output_csv.exists():
        raise RuntimeError(f"Fiji did not write {tag} CSV; exit={proc.returncode}; see {log}")
    values = {}
    with output_csv.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            stem = Path(row["filename"]).stem
            if stem in values:
                raise RuntimeError(f"duplicate Fiji result stem: {stem}")
            geometry = (int(row["width"]), int(row["height"]), int(row["bit_depth"]))
            if geometry != (1024, 1024, 8):
                raise RuntimeError(f"Fiji geometry/bit-depth mismatch: {stem}: {geometry}")
            values[stem] = float(row["IntDen"])
    if set(values) != set(sources):
        raise RuntimeError(f"Fiji IDs mismatch for {tag}: got {len(values)}, expected {len(sources)}")
    print(json.dumps({"fiji": tag, "rows": len(values), "exit_code": proc.returncode}), flush=True)
    return values, output_csv, log, proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Reuse a complete, exact-ID prediction directory from an interrupted postprocess")
    parser.add_argument("--fiji", type=Path, default=DEFAULT_FIJI)
    parser.add_argument("--macro", type=Path, default=DEFAULT_MACRO)
    args = parser.parse_args()

    cfg = read_config(args.config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The comparison run is expected to use the local CUDA device")
    for path in (args.weights, args.manifest, args.config, args.fiji, args.macro):
        if not path.is_file():
            raise FileNotFoundError(path)

    pairs = defaultdict(dict)
    with args.manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"].strip()
            if split not in {"valA", "valB"}:
                continue
            key = (row["source_prefix"].strip(), row["filename"].strip())
            pairs[key][split] = row
    test_pairs = sorted(pairs.items())
    if len(test_pairs) != 1000 or any(set(pair) != {"valA", "valB"} for _, pair in test_pairs):
        raise RuntimeError(f"Expected 1,000 complete official pairs, got {len(test_pairs)}")

    args.output.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output / "predictions"
    pred_dir.mkdir(exist_ok=True)
    expected = {Path(key[1]).stem for key, _ in test_pairs}
    pred = image_map(pred_dir)
    if args.reuse_predictions and set(pred) != expected:
        raise RuntimeError(f"Cannot reuse predictions: got {len(pred)}, expected {len(expected)} exact IDs")
    if pred and not args.reuse_predictions:
        raise RuntimeError("Prediction directory is nonempty; pass --reuse-predictions after verifying IDs")
    config_hash = sha256(args.config)
    weight_hash = sha256(args.weights)
    manifest_hash = sha256(args.manifest)
    model = None
    marker_id = None
    if not args.reuse_predictions:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
        model = StainFlow(cfg).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        marker_id = torch.tensor([cfg.marker_id("HER2")], device=device)
    rows = []
    he_paths, ihc_paths = {}, {}
    started = time.time()

    for index, (key, pair) in enumerate(test_pairs, start=1):
        name = Path(key[1]).stem
        he_path = args.data_root / "valA" / key[0] / key[1]
        ihc_path = args.data_root / "valB" / key[0] / key[1]
        he_paths[name], ihc_paths[name] = he_path, ihc_path
        with Image.open(he_path) as raw:
            he_image = raw.convert("RGB")
        with Image.open(ihc_path) as raw:
            target = np.asarray(raw.convert("RGB"))
        he_rgb = np.asarray(he_image)
        if he_rgb.shape != target.shape or target.shape != (1024, 1024, 3):
            raise RuntimeError(f"Expected paired native 1024 RGB: {name}, {he_rgb.shape}, {target.shape}")

        if not args.reuse_predictions:
            he = torch.from_numpy(he_rgb.copy()).permute(2, 0, 1).float().div(127.5).sub(1)[None].to(device)
            noise = torch.randn(he.shape[1:], generator=torch.Generator().manual_seed(args.seed))[None].to(device)
            with torch.inference_mode(), precision(device, "fp32"):
                generated = sample_rgb(model, he, marker_id, noise, policy=None)
            generated_rgb = np.rint((generated.float().cpu()[0].permute(1, 2, 0).clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
            if generated_rgb.shape != target.shape:
                raise RuntimeError(f"Native output shape mismatch: {name}: {generated_rgb.shape}")
            pred_path = pred_dir / f"{name}.png"
            Image.fromarray(generated_rgb, mode="RGB").save(pred_path)
            pred[name] = pred_path
        with Image.open(pred[name]) as raw:
            generated_rgb = np.asarray(raw.convert("RGB"))
        if generated_rgb.shape != target.shape:
            raise RuntimeError(f"Prediction geometry mismatch: {name}: {generated_rgb.shape}")

        x = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64).ravel()
        y = cv2.cvtColor(he_rgb, cv2.COLOR_RGB2GRAY).astype(np.float64).ravel()
        x -= x.mean()
        y -= y.mean()
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        if denom == 0:
            raise RuntimeError(f"Undefined grayscale correlation: {name}")
        rows.append({
            "stem": name,
            "PSNR": float(peak_signal_noise_ratio(target, generated_rgb, data_range=255)),
            "SSIM": float(structural_similarity(target, generated_rgb, channel_axis=2, data_range=255)),
            "R_c_gray_Pearson": float(np.dot(x, y) / denom),
        })
        if index % 25 == 0 or index == len(test_pairs):
            print(json.dumps({"processed_images": index, "total": 1000,
                              "elapsed_seconds": round(time.time() - started, 1)}), flush=True)

    pred = image_map(pred_dir)
    real = ihc_paths
    if set(pred) != expected or set(real) != expected or set(he_paths) != expected:
        raise RuntimeError("Prediction / paired data IDs do not match the 1,000-image official test list")

    fid_reference_dir = args.output / "fid_reference_valB"
    fid_reference_dir.mkdir(exist_ok=True)
    for name, source in real.items():
        target_path = fid_reference_dir / (name + source.suffix.lower())
        if not target_path.exists() or target_path.stat().st_size != source.stat().st_size:
            target_path.unlink(missing_ok=True)
            try:
                os.link(source, target_path)
            except OSError:
                shutil.copy2(source, target_path)
    fid_real = image_map(fid_reference_dir)
    if set(fid_real) != expected:
        raise RuntimeError(f"FID reference staging mismatch: {len(fid_real)} / {len(expected)}")

    fid_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(json.dumps({"stage": "FID", "device": fid_device}), flush=True)
    fid = float(calculate_fid_given_paths([str(pred_dir), str(fid_reference_dir)],
                                          batch_size=4, device=fid_device, dims=2048, num_workers=0))
    print(json.dumps({"stage": "Fiji", "count_each": 1000}), flush=True)
    real_intden, real_csv, real_log, real_rc = run_fiji("real_valB", real, args.output, args.fiji, args.macro)
    pred_intden, pred_csv, pred_log, pred_rc = run_fiji("generated_valB", pred, args.output, args.fiji, args.macro)
    names = sorted(expected)
    rc = float(np.mean([r["R_c_gray_Pearson"] for r in rows]))
    rp = float(np.corrcoef([real_intden[n] for n in names], [pred_intden[n] for n in names])[0, 1])
    for row in rows:
        row["real_IHC_Fiji_IntDen"] = real_intden[row["stem"]]
        row["generated_IHC_Fiji_IntDen"] = pred_intden[row["stem"]]
    metrics = {
        "PSNR": float(np.mean([r["PSNR"] for r in rows])),
        "SSIM": float(np.mean([r["SSIM"] for r in rows])),
        "FID": fid,
        "R_c": rc,
        "R_p": rp,
        "R_ave": (rc + rp) / 2,
    }
    if not np.isfinite(list(metrics.values())).all():
        raise RuntimeError(f"Nonfinite metric value: {metrics}")

    per_image_path = args.output / "per_image_comparison_metrics.csv"
    with per_image_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "COMPLETE",
        "dataset": "MIST HER2",
        "split": "Official TrainValAB/valA+valB evaluation cohort",
        "image_count": len(rows),
        "checkpoint_stage": "flow",
        "checkpoint_step": 185680,
        "weights_sha256": weight_hash,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "prediction_ids_match": True,
        "native_geometry": "1024x1024 RGB; no resize/crop/flip at inference",
        "inference": "FP32, 4-step Euler, fixed noise seed 1234 per image, no PPO policy",
        "metrics": metrics,
        "fiji": {
            "method": "Direct Fiji macro: Colour Deconvolution H-DAB Colour_2, 8-bit, whole-image IntDen, no threshold/ROI",
            "per_image_intden_columns": ["real_IHC_Fiji_IntDen", "generated_IHC_Fiji_IntDen"],
            "real_rows": len(real_intden), "generated_rows": len(pred_intden),
            "real_exit_code": real_rc, "generated_exit_code": pred_rc,
            "exit_code_note": "Both runs returned code 1; accepted only after 1,000 unique expected IDs and 1024x1024 8-bit rows were validated.",
        },
        "aggregation": {
            "PSNR_SSIM": "mean of per-image RGB metrics, uint8 data_range=255",
            "R_c": "mean of per-image grayscale Pearson correlations between generated IHC and source H&E",
            "R_p": "Pearson correlation of per-image Fiji IntDen values, real IHC versus generated IHC",
            "R_ave": "(R_c + R_p) / 2",
            "FID": "pytorch-fid InceptionV3 2048-d features; batch_size=4",
        },
        "per_image_metrics_csv": per_image_path.name,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out = args.output / "comparison_metrics.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "image_count": len(rows), "metrics": metrics,
                      "summary": str(out), "elapsed_seconds": summary["elapsed_seconds"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
