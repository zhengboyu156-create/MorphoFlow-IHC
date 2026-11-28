"""Paired pilot-dev evaluation for Flow, PPO-on, PPO-off, and zero-action models."""

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morphoflow_ihc.data import IHCCrops
from morphoflow_ihc.engine import CHECKPOINT_FORMAT, open_checkpoint
from morphoflow_ihc.stains import decompose
from morphoflow_ihc.trajectory import (
    corrected_velocity,
    grid,
    policy_state,
    regional_average,
    rgb_to_dab,
    sample_rgb,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def support_metrics(prediction, reference, window, tau):
    ones = torch.ones_like(reference)
    counts = regional_average(ones, window)
    pred = [value / count.clamp_min(1e-8)
            for value, count in zip(regional_average((prediction > tau).float(), window), counts)]
    ref = [value / count.clamp_min(1e-8)
           for value, count in zip(regional_average((reference > tau).float(), window), counts)]
    phase_mae, phase_f1, phase_pred_positive, phase_ref_positive = [], [], [], []
    for pred_fraction, ref_fraction, count in zip(pred, ref, counts):
        valid = count > 0
        phase_mae.append((pred_fraction - ref_fraction).abs()[valid].mean().item())
        predicted_positive = (pred_fraction > 0.01) & valid
        reference_positive = (ref_fraction > 0.01) & valid
        tp = (predicted_positive & reference_positive).sum().float()
        fp = (predicted_positive & ~reference_positive & valid).sum().float()
        fn = (~predicted_positive & reference_positive & valid).sum().float()
        phase_f1.append((2 * tp / (2 * tp + fp + fn).clamp_min(1)).item())
        phase_pred_positive.append(predicted_positive[valid].float().mean().item())
        phase_ref_positive.append(reference_positive[valid].float().mean().item())
    return (
        float(statistics.fmean(phase_mae)),
        float(statistics.fmean(phase_f1)),
        phase_mae,
        phase_f1,
        float(statistics.fmean(phase_pred_positive)),
        float(statistics.fmean(phase_ref_positive)),
        phase_pred_positive,
        phase_ref_positive,
    )


def prefix_bootstrap(rows, model_name, seed, draws=10000):
    grouped = {}
    for row in rows:
        if row["model"] == model_name:
            grouped.setdefault(row["source_prefix"], []).append(float(row["value"]))
    prefix_means = np.asarray([statistics.fmean(values) for _, values in sorted(grouped.items())], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(prefix_means), size=(draws, len(prefix_means)))
    bootstrap_means = prefix_means[samples].mean(axis=1)
    return {
        "unit": "source_prefix (not verified patient identity); equal weight per prefix",
        "prefix_count": len(prefix_means),
        "pair_count": sum(len(values) for values in grouped.values()),
        "mean_of_prefix_means": float(prefix_means.mean()),
        "descriptive_95pct_percentile_interval": [float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])],
        "prefixes_delta_negative": int((prefix_means < 0).sum()),
        "prefixes_delta_positive": int((prefix_means > 0).sum()),
        "prefixes_delta_zero": int((prefix_means == 0).sum()),
        "bootstrap_draws": draws,
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--ppo", type=Path, required=True)
    parser.add_argument("--zero-action", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261130)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing pilot evaluation: {args.output}")
    device = torch.device(args.device)
    flow_model, _, flow_state = open_checkpoint(args.flow, device)
    ppo_model, ppo_policy, ppo_state = open_checkpoint(args.ppo, device)
    zero_model, _, zero_state = open_checkpoint(args.zero_action, device)
    if flow_state["step"] != 1000 or ppo_state["step"] != 1000 or zero_state["step"] != 1000:
        raise ValueError("Expected the 1,000-update pilot checkpoint triplet")
    if ppo_policy is None or ppo_state.get("policy_objective") != "ppo":
        raise ValueError("PPO checkpoint lacks its trained policy")
    for state in (flow_state, ppo_state, zero_state):
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("Pilot evaluation only accepts current-format checkpoints")
    cfg = flow_model.cfg
    # The policy-scale follow-up intentionally changes only PPO's velocity limit.
    # Require every other model/loss setting to match the Flow and zero-action arms.
    expected_cfg = vars(cfg).copy()
    ppo_cfg = vars(ppo_model.cfg).copy()
    ppo_velocity_limit = ppo_cfg.pop("velocity_limit", None)
    expected_without_velocity = expected_cfg.copy()
    expected_without_velocity.pop("velocity_limit", None)
    if ppo_cfg != expected_without_velocity or zero_model.cfg != cfg:
        raise ValueError("Pilot checkpoints differ in settings other than PPO velocity_limit")
    dataset = IHCCrops(args.cache, cfg)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    ids = split["pilot_dev_pair_ids"]
    if len(ids) != 128 or len(set(ids)) != 128:
        raise ValueError("Expected the fixed unique 128-pair pilot-dev list")
    indices = {entry["pair_id"]: index for index, entry in enumerate(dataset.entries)}
    if set(ids) - set(indices):
        raise ValueError("Pilot-dev list and cached pairs do not match")

    models = {
        "flow1000": (flow_model, None),
        "ppo_policy_on": (ppo_model, ppo_policy),
        "ppo_policy_off_same_checkpoint": (ppo_model, None),
        "zero_action1000": (zero_model, None),
    }
    rows = []
    marker_id = torch.tensor([cfg.marker_id("HER2")], device=device)
    tau = cfg.markers["HER2"]
    for index, pair_id in enumerate(sorted(ids)):
        digest = hashlib.sha256(f"{args.seed}:{pair_id}".encode()).digest()
        sample_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
        marker, target_image, he = dataset[(indices[pair_id], sample_seed)]
        if marker != int(marker_id.item()):
            raise ValueError(f"Unexpected marker for {pair_id}")
        he = he.unsqueeze(0).to(device)
        target_image = target_image.unsqueeze(0).to(device)
        cpu_generator = torch.Generator().manual_seed(sample_seed)
        noise = torch.randn(he.shape, generator=cpu_generator).to(device)
        reference_dab = rgb_to_dab(target_image, cfg)
        target_u8 = np.rint((target_image[0].cpu().permute(1, 2, 0).numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
        _, _, tissue = decompose(target_u8, cfg, "IHC")
        tissue_tensor = torch.from_numpy(tissue.copy())[None, None].to(device)
        target_total = (reference_dab[0, 0] * tissue_tensor[0, 0]).sum().item()
        per_model = {}
        generated_by_model = {}
        with torch.inference_mode():
            for name, (model, policy) in models.items():
                generated = sample_rgb(model, he, marker_id, noise, policy)
                generated_by_model[name] = generated
                predicted_dab = rgb_to_dab(generated, cfg)
                metrics = {}
                for window in (64, 128):
                    (mae, f1, phase_mae, phase_f1, predicted_positive_fraction,
                     reference_positive_fraction, phase_predicted_positive_fraction,
                     phase_reference_positive_fraction) = support_metrics(predicted_dab, reference_dab, window, tau)
                    metrics[f"support_mae_{window}"] = mae
                    metrics[f"support_f1_{window}"] = f1
                    metrics[f"support_pred_positive_window_fraction_{window}"] = predicted_positive_fraction
                    metrics[f"support_ref_positive_window_fraction_{window}"] = reference_positive_fraction
                    for phase, value in enumerate(phase_mae):
                        metrics[f"support_mae_{window}_phase{phase}"] = value
                    for phase, value in enumerate(phase_f1):
                        metrics[f"support_f1_{window}_phase{phase}"] = value
                    for phase, value in enumerate(phase_predicted_positive_fraction):
                        metrics[f"support_pred_positive_window_fraction_{window}_phase{phase}"] = value
                    for phase, value in enumerate(phase_reference_positive_fraction):
                        metrics[f"support_ref_positive_window_fraction_{window}_phase{phase}"] = value
                predicted_total = (predicted_dab[0, 0] * tissue_tensor[0, 0]).sum().item()
                metrics["dab_total_relative_error"] = abs(predicted_total - target_total) / max(target_total, 1e-8)
                metrics["dab_mean_on_reference_tissue"] = (predicted_dab[0, 0] * tissue_tensor[0, 0]).sum().item() / max(int(tissue_tensor.sum().item()), 1)
                metrics["output_rgb01_mean"] = ((generated.float() + 1) / 2).mean().item()
                per_model[name] = metrics
                rows.append({"pair_id": pair_id, "source_prefix": pair_id.split("/", 1)[0],
                             "crop_seed": sample_seed, "model": name, **metrics})
            times = grid(ppo_model, noise)
            _, base_velocity, observation = policy_state(ppo_model, ppo_policy, he, marker_id, noise, times)
            policy_mean = ppo_policy(observation)
            policy_correction = corrected_velocity(ppo_model, base_velocity, policy_mean) - base_velocity
            policy_diagnostics = {
                "policy_action_mean_rms": policy_mean.square().mean().sqrt().item(),
                "policy_action_mean_abs": policy_mean.abs().mean().item(),
                "policy_velocity_correction_rms": policy_correction.square().mean().sqrt().item(),
                "policy_correction_to_base_rms": (policy_correction.square().mean().sqrt()
                                                   / base_velocity.square().mean().sqrt().clamp_min(1e-12)).item(),
            }
            per_model["ppo_policy_on"].update(policy_diagnostics)
            for row in rows[-len(models):]:
                if row["pair_id"] == pair_id and row["model"] == "ppo_policy_on":
                    row.update(policy_diagnostics)
                    break
        baseline = per_model["zero_action1000"]
        ppo_on = per_model["ppo_policy_on"]
        ppo_off = per_model["ppo_policy_off_same_checkpoint"]
        for window in (64, 128):
            for metric in (f"support_mae_{window}", f"support_f1_{window}"):
                rows.append({"pair_id": pair_id, "source_prefix": pair_id.split("/", 1)[0],
                             "crop_seed": sample_seed, "model": f"paired_delta_ppo_on_minus_zero_{metric}",
                             "value": ppo_on[metric] - baseline[metric]})
                rows.append({"pair_id": pair_id, "source_prefix": pair_id.split("/", 1)[0],
                             "crop_seed": sample_seed, "model": f"paired_delta_ppo_on_minus_same_model_off_{metric}",
                             "value": ppo_on[metric] - ppo_off[metric]})
        rows.append({"pair_id": pair_id, "source_prefix": pair_id.split("/", 1)[0],
                     "crop_seed": sample_seed, "model": "paired_delta_ppo_on_minus_zero_dab_total_relative_error",
                     "value": ppo_on["dab_total_relative_error"] - baseline["dab_total_relative_error"]})
        rows.append({"pair_id": pair_id, "source_prefix": pair_id.split("/", 1)[0],
                     "crop_seed": sample_seed, "model": "paired_delta_ppo_on_minus_same_model_off_dab_total_relative_error",
                     "value": ppo_on["dab_total_relative_error"] - ppo_off["dab_total_relative_error"]})
        for comparison, left, right in (
            ("ppo_on_minus_zero", "ppo_policy_on", "zero_action1000"),
            ("ppo_on_minus_same_model_off", "ppo_policy_on", "ppo_policy_off_same_checkpoint"),
        ):
            rgb_delta = (generated_by_model[left].float() - generated_by_model[right].float()).abs().mean().item() / 2
            rows.append({"pair_id": pair_id, "source_prefix": pair_id.split("/", 1)[0],
                         "crop_seed": sample_seed, "model": f"paired_output_delta_{comparison}_rgb01_mae",
                         "value": rgb_delta})
        if (index + 1) % 16 == 0 or index + 1 == len(ids):
            print(json.dumps({"evaluated_pairs": index + 1, "total": len(ids)}), flush=True)

    args.output.mkdir(parents=True)
    csv_path = args.output / "per_pair_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)

    aggregate = {}
    for model_name in models:
        mode_rows = [row for row in rows if row["model"] == model_name]
        fields = [key for key in mode_rows[0] if key not in {"pair_id", "source_prefix", "crop_seed", "model"}]
        aggregate[model_name] = {field: {
            "mean": statistics.fmean(float(row[field]) for row in mode_rows),
            "median": statistics.median(float(row[field]) for row in mode_rows),
        } for field in fields}
    deltas = {}
    for label in ("paired_delta_ppo_on_minus_zero_", "paired_delta_ppo_on_minus_same_model_off_"):
        deltas[label] = {}
        for metric in ("support_mae_64", "support_f1_64", "support_mae_128", "support_f1_128"):
            values = [float(row["value"]) for row in rows if row["model"] == label + metric]
            deltas[label][metric] = {"mean": statistics.fmean(values), "median": statistics.median(values)}
    total_delta = [float(row["value"]) for row in rows
                   if row["model"] == "paired_delta_ppo_on_minus_zero_dab_total_relative_error"]
    deltas["ppo_on_minus_zero_dab_total_relative_error"] = {
        "mean": statistics.fmean(total_delta), "median": statistics.median(total_delta)}
    for label in ("paired_delta_ppo_on_minus_same_model_off_dab_total_relative_error",
                  "paired_output_delta_ppo_on_minus_zero_rgb01_mae",
                  "paired_output_delta_ppo_on_minus_same_model_off_rgb01_mae"):
        values = [float(row["value"]) for row in rows if row["model"] == label]
        deltas[label] = {"mean": statistics.fmean(values), "median": statistics.median(values)}
    by_prefix = {}
    for prefix in sorted({row["source_prefix"] for row in rows}):
        relevant = [row for row in rows if row["source_prefix"] == prefix
                    and row["model"] == "paired_delta_ppo_on_minus_zero_support_mae_128"]
        by_prefix[prefix] = {"pairs": len(relevant), "mean_delta_support_mae_128":
                             statistics.fmean(float(row["value"]) for row in relevant)}

    summary = {
        "evaluation_only_no_weight_updates": True,
        "pilot_dev_pair_count": len(ids),
        "source_prefix_count": len({pair_id.split("/", 1)[0] for pair_id in ids}),
        "grouping_caveat": "source_prefix coverage; not a verified patient identity",
        "cache_index_sha256": dataset.fingerprint,
        "split_sha256": sha256(args.split),
        "crop_and_noise": "one deterministic paired 512 crop and one common Gaussian noise tensor per pair across all models",
        "precision": "FP32",
        "checkpoints": {"flow1000": {"sha256": sha256(args.flow), "step": flow_state["step"]},
                        "ppo1000": {"sha256": sha256(args.ppo), "step": ppo_state["step"]},
                        "zero_action1000": {"sha256": sha256(args.zero_action), "step": zero_state["step"]}},
        "policy_velocity_limit": ppo_velocity_limit,
        "metric_definitions": {
            "support_f1_window_fraction_threshold": 0.01,
            "support_f1_role": "diagnostic only; interpret with predicted/reference positive-window prevalence",
            "support_mae_role": "continuous primary regional support metric",
        },
        "metrics": aggregate,
        "paired_deltas": deltas,
        "descriptive_prefix_bootstrap": {
            metric: prefix_bootstrap(rows, metric, args.seed + offset)
            for offset, metric in enumerate((
                "paired_delta_ppo_on_minus_zero_support_mae_64",
                "paired_delta_ppo_on_minus_zero_support_mae_128",
                "paired_delta_ppo_on_minus_zero_dab_total_relative_error",
                "paired_delta_ppo_on_minus_same_model_off_support_mae_128",
                "paired_delta_ppo_on_minus_same_model_off_dab_total_relative_error",
            ))
        },
        "per_prefix_ppo_minus_zero_support_mae_128": by_prefix,
        "csv_sha256": sha256(csv_path),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device)},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
