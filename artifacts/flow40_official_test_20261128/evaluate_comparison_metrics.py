"""Evaluate Flow and joint checkpoints with the MIST HER2 comparison protocol."""

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from tempfile import TemporaryDirectory
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from pytorch_fid.fid_score import IMAGE_EXTENSIONS, calculate_fid_given_paths
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from morphoflow_ihc.config import read_config
from morphoflow_ihc.data import atomic_write, sha256
from mist_evaluation import checkpoint_for_evaluation, image_seed, official_pairs, predict_rgb, read_pair


DEFAULT_FIJI = Path(r"D:\Fiji\Fiji\fiji-windows-x64.exe")
DEFAULT_MACRO = Path(__file__).resolve().with_name("direct_fiji_intden.ijm")
IMAGE_SUFFIXES = {f".{extension}" for extension in IMAGE_EXTENSIONS} | {".tiff"}


def stage_images(sources, input_dir):
    """Refresh staged files and verify bytes; never trust size-only cache hits."""
    input_dir.mkdir(parents=True, exist_ok=True)
    expected = set()
    for stem, src in sources.items():
        dst = input_dir / (stem + src.suffix.lower())
        expected.add(dst.name)
        dst.unlink(missing_ok=True)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        if sha256(dst) != sha256(src):
            raise RuntimeError(f"Staged image hash mismatch: {src} -> {dst}")
    for path in input_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES \
                and path.name not in expected:
            path.unlink()


def image_map(folder):
    result = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            if path.stem in result:
                raise RuntimeError(f"duplicate image stem: {path.stem} in {folder}")
            result[path.stem] = path
    return result


def run_fiji(tag, sources, root, executable, macro_source, work_root):
    work_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="mist-", dir=work_root) as temporary:
        return run_fiji_staged(tag, sources, root, executable, macro_source, Path(temporary))


def run_fiji_staged(tag, sources, root, executable, macro_source, stage):
    input_dir = stage / "images"
    stage_images(sources, input_dir)
    output_csv = stage / "intden.csv"
    macro = stage / "batch.ijm"
    shutil.copy2(macro_source, macro)
    archived_csv = root / f"{tag}_intden.csv"
    archived_log = root / f"{tag}_fiji.log"
    command = [str(executable.resolve()), "--jaunch-skip-console-check", "--allow-multiple",
               "-macro", str(macro), f"{input_dir},{output_csv}"]
    with archived_log.open("w", encoding="utf-8", errors="replace") as stream:
        proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=3 * 60 * 60)
    done = output_csv.with_suffix(".csv.done")
    if not done.is_file() or done.read_text(encoding="utf-8").strip() != str(len(sources)):
        raise RuntimeError(f"Fiji did not finish {tag}; exit={proc.returncode}; see {archived_log}")
    if not output_csv.exists():
        raise RuntimeError(f"Fiji did not write {tag} CSV; exit={proc.returncode}; see {archived_log}")
    values = {}
    with output_csv.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            stem = Path(row["filename"]).stem
            if stem in values:
                raise RuntimeError(f"duplicate Fiji result stem: {stem}")
            geometry = (int(row["width"]), int(row["height"]), int(row["bit_depth"]))
            if geometry != (1024, 1024, 8):
                raise RuntimeError(f"Fiji geometry/bit-depth mismatch: {stem}: {geometry}")
            value = float(row["IntDen"])
            if float(row["area"]) != 1024 ** 2 or not 0 <= value <= 255 * 1024 ** 2:
                raise RuntimeError(f"Invalid whole-image Fiji intensity: {stem}")
            values[stem] = value
    if set(values) != set(sources):
        raise RuntimeError(f"Fiji IDs mismatch for {tag}: got {len(values)}, expected {len(sources)}")
    shutil.copy2(output_csv, archived_csv)
    if proc.returncode != 0:
        print(json.dumps({"warning": "Fiji exited nonzero after validating the completion file and CSV",
                          "fiji": tag, "exit_code": proc.returncode}), flush=True)
    return values, archived_csv, archived_log, proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--checkpoint-step", type=int,
                        help="Training step from the checkpoint provenance; read from a full checkpoint when available")
    parser.add_argument("--reuse-predictions", action="store_true",
                        help="Reuse a complete, exact-ID prediction directory from an interrupted postprocess")
    parser.add_argument("--fiji", type=Path, default=DEFAULT_FIJI)
    parser.add_argument("--macro", type=Path, default=DEFAULT_MACRO)
    parser.add_argument("--allow-historical-weights", action="store_true")
    parser.add_argument("--disable-policy", action="store_true", help="Ablate the policy on the same joint-trained generator")
    parser.add_argument("--fiji-work-root", type=Path, default=Path(__file__).resolve().parent / "fiji_ascii")
    args = parser.parse_args()

    cfg = read_config(args.config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The comparison run is expected to use the local CUDA device")
    for path in (args.weights, args.manifest, args.config, args.fiji, args.macro):
        if not path.is_file():
            raise FileNotFoundError(path)

    test_pairs = official_pairs(args.manifest)
    expected = {Path(key[1]).stem for key, _ in test_pairs}
    args.fiji_work_root = args.fiji_work_root.resolve()
    if not str(args.fiji_work_root).isascii() or "," in str(args.fiji_work_root):
        raise ValueError("Choose an ASCII Fiji work directory with no commas")

    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "comparison_metrics_summary.json").exists():
        raise ValueError("Choose a new directory for a completed comparison run")
    pred_dir = args.output / "predictions"
    pred_dir.mkdir(exist_ok=True)
    pred = image_map(pred_dir)
    config_hash = sha256(args.config)
    weight_hash = sha256(args.weights)
    manifest_hash = sha256(args.manifest)
    model, policy, checkpoint = checkpoint_for_evaluation(
        args, cfg, device, load_model=not args.reuse_predictions,
    )
    inference = f"native 1024 RGB, FP32, {cfg.ode_steps}-step Euler, " + (
        "deterministic policy mean" if checkpoint["policy_enabled"] else "no policy"
    )
    prediction_meta_path = pred_dir / "prediction_provenance.json"
    expected_prediction_meta = {
        "weights_sha256": weight_hash,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        **checkpoint,
        "base_seed": args.seed,
        "seed_policy": "sha256(base_seed, source_prefix/filename), first 63 bits",
        "inference": inference,
        "inference_code_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): sha256(path)
            for path in (Path(__file__).resolve(), Path(__file__).resolve().parent / "mist_evaluation.py",
                         *sorted((Path(__file__).resolve().parents[2] / "morphoflow_ihc").glob("*.py")))
        },
        "runtime": {"torch": str(torch.__version__), "device": torch.cuda.get_device_name(device)},
    }
    if args.reuse_predictions:
        if set(pred) != expected:
            raise RuntimeError(f"Cannot reuse predictions: got {len(pred)}, expected {len(expected)} exact IDs")
        if any(path.suffix != ".png" for path in pred.values()):
            raise RuntimeError("Prediction reuse requires PNG files")
        if not prediction_meta_path.is_file():
            raise RuntimeError("Prediction provenance is missing; regenerate predictions with this evaluator")
        prediction_meta = json.loads(prediction_meta_path.read_text(encoding="utf-8"))
        for key, value in expected_prediction_meta.items():
            if prediction_meta.get(key) != value:
                raise RuntimeError(f"Prediction provenance mismatch for {key}; regenerate predictions")
        recorded_hashes = prediction_meta.get("prediction_sha256", {})
        if set(recorded_hashes) != expected or any(sha256(pred[name]) != recorded_hashes[name] for name in expected):
            raise RuntimeError("Prediction file hashes do not match provenance; regenerate predictions")
    elif pred:
        raise RuntimeError("Prediction directory is nonempty; pass --reuse-predictions with matching provenance")
    marker_id = torch.tensor([cfg.marker_id("HER2")], device=device)
    rows = []
    he_paths, ihc_paths = {}, {}
    started = time.time()

    for index, (key, pair) in enumerate(test_pairs, start=1):
        name = Path(key[1]).stem
        (he_path, ihc_path), (he_rgb, target) = read_pair(args.data_root, key, pair, cfg)
        if not args.reuse_predictions:
            generated_rgb = predict_rgb(model, policy, he_rgb, marker_id, image_seed(args.seed, key))
            if generated_rgb.shape != target.shape:
                raise RuntimeError(f"Native output shape mismatch: {name}: {generated_rgb.shape}")
            pred_path = pred_dir / f"{name}.png"
            atomic_write(pred_path, lambda stream: Image.fromarray(generated_rgb).save(stream, format="PNG"))
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
            "pair_id": f"{key[0]}/{key[1]}",
            "noise_seed": image_seed(args.seed, key),
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
    if not args.reuse_predictions:
        prediction_meta = {
            **expected_prediction_meta,
            "prediction_sha256": {name: sha256(pred[name]) for name in sorted(expected)},
        }
        atomic_write(prediction_meta_path,
                     lambda stream: stream.write((json.dumps(prediction_meta, indent=2) + "\n").encode("utf-8")))

    fid_reference_dir = args.output / "fid_reference_valB"
    stage_images(real, fid_reference_dir)
    fid_real = image_map(fid_reference_dir)
    if set(fid_real) != expected:
        raise RuntimeError(f"FID reference staging mismatch: {len(fid_real)} / {len(expected)}")

    fid_device = str(device)
    print(json.dumps({"stage": "FID", "device": fid_device}), flush=True)
    fid = float(calculate_fid_given_paths([str(pred_dir), str(fid_reference_dir)],
                                          batch_size=4, device=fid_device, dims=2048, num_workers=0))
    print(json.dumps({"stage": "Fiji", "count_each": 1000}), flush=True)
    real_intden, real_csv, real_log, real_rc = run_fiji("real_valB", real, args.output, args.fiji, args.macro, args.fiji_work_root)
    pred_intden, pred_csv, pred_log, pred_rc = run_fiji("generated_valB", pred, args.output, args.fiji, args.macro, args.fiji_work_root)
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
        **checkpoint,
        "weights_sha256": weight_hash,
        "config_sha256": config_hash,
        "manifest_sha256": manifest_hash,
        "prediction_ids_match": True,
        "native_geometry": "1024x1024 RGB; no resize/crop/flip at inference",
        "inference": inference,
        "prediction_provenance_sha256": sha256(prediction_meta_path),
        "noise_seed_policy": "sha256(base_seed, source_prefix/filename), first 63 bits; recorded per image",
        "metrics": metrics,
        "fiji": {
            "method": "Direct Fiji macro: Colour Deconvolution H-DAB Colour_2, 8-bit, whole-image IntDen, no threshold/ROI",
            "per_image_intden_columns": ["real_IHC_Fiji_IntDen", "generated_IHC_Fiji_IntDen"],
            "real_rows": len(real_intden), "generated_rows": len(pred_intden),
            "real_exit_code": real_rc, "generated_exit_code": pred_rc,
            "macro_sha256": sha256(args.macro),
            "launcher_sha256": sha256(args.fiji),
            "completion_marker_verified": True,
            "exit_code_note": "Each Fiji run must write its expected DONE count file and pass exact-ID, 1024x1024, 8-bit CSV validation; nonzero launcher exit is recorded as a warning.",
        },
        "aggregation": {
            "PSNR_SSIM": "mean of per-image RGB metrics, uint8 data_range=255",
            "R_c": "mean of per-image grayscale Pearson correlations between generated IHC and source H&E; proxy only",
            "R_p": "Pearson correlation of per-image Fiji IntDen values, real IHC versus generated IHC; proxy only",
            "R_ave": "(R_c + R_p) / 2; nonstandard proxy, not a calibrated clinical score",
            "FID": "pytorch-fid InceptionV3 2048-d features; batch_size=4",
        },
        "per_image_metrics_csv": per_image_path.name,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    out = args.output / "comparison_metrics_summary.json"
    atomic_write(out, lambda stream: stream.write((json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")))
    print(json.dumps({"status": "COMPLETE", "image_count": len(rows), "metrics": metrics,
                      "summary": str(out), "elapsed_seconds": summary["elapsed_seconds"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
