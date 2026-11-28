"""Evaluate regional DAB support error on official MIST HER2 test pairs."""

import argparse
import csv
import json
import time
from pathlib import Path
import sys

import numpy as np
import torch

# This workstation has another MorphoFlow checkout earlier in PYTHONPATH.
# Pin imports to the exact checkout whose weights and config are being tested.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from morphoflow_ihc.config import read_config
from morphoflow_ihc.data import atomic_write, sha256
from morphoflow_ihc.engine import choose_device
from morphoflow_ihc.stains import decompose
from morphoflow_ihc.trajectory import regional_average
from mist_evaluation import checkpoint_for_evaluation, image_seed, official_pairs, predict_rgb, read_pair


def support_fraction_mae(estimate, actual, tissue, threshold, window):
    if not tissue.any():
        return None
    estimate_t, actual_t = (
        torch.from_numpy(array.astype(np.float32))[None, None] for array in (estimate, actual)
    )
    tissue_t = torch.from_numpy(tissue.astype(np.bool_))[None, None]
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
    parser.add_argument("--allow-historical-weights", action="store_true")
    parser.add_argument("--disable-policy", action="store_true", help="Ablate the policy on the same joint-trained generator")
    args = parser.parse_args()

    cfg = read_config(args.config)
    device = choose_device(args.device)
    test_pairs = official_pairs(args.manifest)
    model, policy, checkpoint = checkpoint_for_evaluation(args, cfg, device)
    weights_hash = sha256(args.weights)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "per_image_metrics.csv"
    summary_path = args.output / "summary.json"
    if csv_path.exists() or summary_path.exists():
        raise ValueError("Choose a new output directory for the regional evaluation")
    rows = []
    started = time.time()
    marker_id = torch.tensor([cfg.marker_id("HER2")], device=device)
    tau = cfg.markers["HER2"]
    for index, (key, pair) in enumerate(test_pairs, start=1):
        _, (he_rgb, target_rgb) = read_pair(args.data_root, key, pair, cfg)
        sample_seed = image_seed(args.seed, key)
        generated_rgb = predict_rgb(model, policy, he_rgb, marker_id, sample_seed)
        _, predicted_dab, _ = decompose(generated_rgb, cfg, "IHC")
        _, reference_dab, tissue = decompose(target_rgb, cfg, "IHC")
        if predicted_dab.shape != reference_dab.shape:
            raise ValueError(f"Prediction/reference shape mismatch: {key}")

        rows.append({
            "pair_id": f"{key[0]}/{key[1]}",
            "noise_seed": sample_seed,
            "region_support_fraction_mae": support_fraction_mae(
                predicted_dab, reference_dab, tissue, tau, cfg.region
            ),
            "reward_region_support_fraction_mae": support_fraction_mae(
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
        **checkpoint,
        "weights_sha256": weights_hash,
        "config_sha256": sha256(args.config),
        "manifest_sha256": sha256(args.manifest),
        "test_split": "official MIST HER2 valA/valB (1,000 pairs)",
        "test_pair_count": len(rows),
        "base_seed": args.seed,
        "noise_seed_policy": "sha256(base_seed, source_prefix/filename), first 63 bits; seed recorded per image",
        "inference": f"native 1024 RGB, FP32, {cfg.ode_steps}-step Euler",
        "policy_action": "deterministic mean" if policy is not None else None,
        "region_px": cfg.region,
        "reward_region_px": cfg.reward_region,
        "metrics": {},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    for column in ("region_support_fraction_mae",
                   "reward_region_support_fraction_mae",
                   "reference_tissue_fraction"):
        values = np.asarray([row[column] for row in rows if row[column] is not None], dtype=np.float64)
        summary["metrics"][column] = {
            "count": len(values),
            "mean": float(values.mean()) if len(values) else None,
            "median": float(np.median(values)) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        }
    atomic_write(summary_path, lambda stream: stream.write((json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
