"""Evaluate a bare Flow-stage model state dict on official MIST HER2 test pairs."""

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

# This workstation has another MorphoFlow checkout earlier in PYTHONPATH.
# Pin imports to the exact checkout whose weights and config are being tested.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from morphoflow_ihc.config import Config, read_config
from morphoflow_ihc.engine import precision
from morphoflow_ihc.model import StainFlow
from morphoflow_ihc.stains import canonical_size, decompose
from morphoflow_ihc.trajectory import regional_average, sample_rgb


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_seed(base_seed, pair_key):
    """Stable, distinct seed for each official paired image."""
    token = f"{int(base_seed)}:{pair_key[0]}/{pair_key[1]}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & ((1 << 63) - 1)


def support_fraction_mae(estimate, actual, tissue, threshold, window):
    height, width = estimate.shape
    pad = (0, -width % window, 0, -height % window)
    estimate_t, actual_t = (
        F.pad(torch.from_numpy(array.astype(np.float32))[None, None], pad)
        for array in (estimate, actual)
    )
    tissue_t = F.pad(torch.from_numpy(tissue.astype(np.bool_))[None, None], pad)
    counts = regional_average(tissue_t.float(), window)
    supports = [regional_average(((value > threshold) & tissue_t).float(), window)
                for value in (estimate_t, actual_t)]
    errors = []
    for count, pred_support, ref_support in zip(counts, supports[0], supports[1]):
        pred_fraction = pred_support / count.clamp_min(1e-8)
        ref_fraction = ref_support / count.clamp_min(1e-8)
        valid = count > 0
        errors.append((pred_fraction - ref_fraction).abs()[valid].mean())
    return torch.stack(errors).mean().item()


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
                        help="Step from checkpoint metadata; required to identify a raw state-dict checkpoint")
    args = parser.parse_args()

    cfg = read_config(args.config)
    device = torch.device(args.device)
    weights_hash = sha256(args.weights)
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        checkpoint_step = state.get("step", args.checkpoint_step)
        if state.get("stage") not in (None, "flow"):
            raise RuntimeError(f"Expected a Flow checkpoint, got stage={state.get('stage')!r}")
        if args.checkpoint_step is not None and checkpoint_step != args.checkpoint_step:
            raise RuntimeError("--checkpoint-step does not match checkpoint metadata")
        checkpoint_config = state.get("config")
        if checkpoint_config is not None and Config(**checkpoint_config) != cfg:
            raise RuntimeError("Evaluation config does not match checkpoint's embedded config")
        model_state = state["model"]
    else:
        checkpoint_step = args.checkpoint_step
        model_state = state
    if checkpoint_step is None:
        raise RuntimeError("Checkpoint step is unavailable; pass --checkpoint-step from its training provenance")
    model = StainFlow(cfg).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

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
        raise RuntimeError(f"Expected 1,000 complete official test pairs; got {len(test_pairs)}")
    if len({Path(key[1]).stem for key, _ in test_pairs}) != len(test_pairs):
        raise RuntimeError("Official pairs contain duplicate filename stems; result IDs would collide")

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "per_image_metrics.csv"
    summary_path = args.output / "summary.json"
    rows = []
    started = time.time()
    marker_id = torch.tensor([cfg.marker_id("HER2")], device=device)
    tau = cfg.markers["HER2"]
    factor = 2 ** (len(cfg.channel_mult) - 1)

    for index, (key, pair) in enumerate(test_pairs, start=1):
        he_row, ihc_row = pair["valA"], pair["valB"]
        he_path = args.data_root / "valA" / he_row["source_prefix"] / he_row["filename"]
        ihc_path = args.data_root / "valB" / ihc_row["source_prefix"] / ihc_row["filename"]
        with Image.open(he_path) as raw:
            he_image = raw.convert("RGB")
        with Image.open(ihc_path) as raw:
            ref_image = raw.convert("RGB")
        mpp = float(he_row.get("mpp") or cfg.pixel_size_um)
        target_size = canonical_size(he_image.size, mpp, cfg)
        if target_size != he_image.size:
            he_image = he_image.resize(target_size, Image.Resampling.BILINEAR)
        if target_size[0] % factor or target_size[1] % factor:
            raise ValueError(f"Unsupported inference size {target_size} for {key}")

        he_rgb = np.asarray(he_image)
        he = torch.from_numpy(he_rgb.copy()).permute(2, 0, 1).float().div(127.5).sub(1).unsqueeze(0).to(device)
        sample_seed = image_seed(args.seed, key)
        noise = torch.randn(he.shape[1:], generator=torch.Generator().manual_seed(sample_seed))[None].to(device)
        with torch.inference_mode(), precision(device, "fp32"):
            generated = sample_rgb(model, he, marker_id, noise, policy=None)
        generated_rgb = np.rint((generated.float().cpu()[0].permute(1, 2, 0).clamp(-1, 1).numpy() + 1) * 127.5).astype(np.uint8)
        if generated_rgb.shape[:2] != ref_image.size[::-1]:
            generated_rgb = np.asarray(Image.fromarray(generated_rgb).resize(ref_image.size, Image.Resampling.BILINEAR))
        _, predicted_dab, _ = decompose(generated_rgb, cfg, "IHC")
        _, reference_dab, tissue = decompose(np.asarray(ref_image), cfg, "IHC")
        if predicted_dab.shape != reference_dab.shape:
            raise ValueError(f"Prediction/reference shape mismatch: {key}")

        rows.append({
            "pair_id": f"{key[0]}/{key[1]}",
            "noise_seed": sample_seed,
            "region_support_fraction_mae_32px": support_fraction_mae(
                predicted_dab, reference_dab, tissue, tau, cfg.region
            ),
            "reward_region_support_fraction_mae_128px": support_fraction_mae(
                predicted_dab, reference_dab, tissue, tau, cfg.reward_region
            ),
            "reference_tissue_fraction": float(tissue.mean()),
        })

        if index % 25 == 0 or index == len(test_pairs):
            with csv_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps({"evaluated": index, "total": len(test_pairs),
                              "elapsed_seconds": round(time.time() - started, 1)}), flush=True)

    summary = {
        "checkpoint_stage": "flow",
        "checkpoint_step": checkpoint_step,
        "weights_sha256": weights_hash,
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(args.manifest),
        "test_split": "official MIST HER2 valA/valB (1,000 pairs)",
        "test_pair_count": len(rows),
        "base_seed": args.seed,
        "noise_seed_policy": "sha256(base_seed, source_prefix/filename), first 63 bits; seed recorded per image",
        "inference": f"native-size full-image, {cfg.ode_steps}-step Euler, FP32, no PPO policy",
        "metrics": {},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    for column in ("region_support_fraction_mae_32px",
                   "reward_region_support_fraction_mae_128px",
                   "reference_tissue_fraction"):
        values = np.asarray([row[column] for row in rows], dtype=np.float64)
        summary["metrics"][column] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "std": float(values.std(ddof=1)),
        }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
