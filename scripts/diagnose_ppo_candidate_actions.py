"""Measure candidate-action, velocity, output and reward sensitivity without updates."""

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# This workstation has other MorphoFlow worktrees earlier in PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morphoflow_ihc.config import read_config
from morphoflow_ihc.data import IHCCrops
from morphoflow_ihc.model import StainFlow, VelocityPolicy
from morphoflow_ihc.trajectory import (
    action_distribution,
    corrected_velocity,
    grid,
    integrate,
    paired_expression_reward_components,
    policy_state,
    reward_reference,
    rgb_to_dab,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_states(dataset, pair_ids, count):
    entries = {entry["pair_id"]: index for index, entry in enumerate(dataset.entries)}
    by_prefix = {}
    for pair_id in sorted(pair_ids):
        prefix = pair_id.split("/", 1)[0]
        by_prefix.setdefault(prefix, pair_id)
    selected = [by_prefix[prefix] for prefix in sorted(by_prefix)]
    if count > len(selected):
        raise ValueError(f"Requested {count} states but dev has only {len(selected)} source prefixes")
    selected = selected[:count]
    return [(pair_id, entries[pair_id]) for pair_id in selected]


def autocast_context(device, mode):
    if mode == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", enabled=False)


def diagnose_one(model, policy, dataset, pair_id, dataset_index, *, device, candidates, seed):
    digest = hashlib.sha256(f"{seed}:{pair_id}".encode()).digest()
    crop_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    marker_value, reference, he = dataset[(dataset_index, crop_seed)]
    marker = torch.tensor([marker_value], device=device)
    reference = reference.unsqueeze(0).to(device)
    he = he.unsqueeze(0).to(device)

    torch.manual_seed(crop_seed)
    torch.cuda.manual_seed_all(crop_seed)
    noise = torch.randn_like(he)
    epsilon_seed = (crop_seed + 1709) % (2**63 - 1)
    generator = torch.Generator(device=device).manual_seed(epsilon_seed)
    action_epsilon = torch.randn(
        (candidates, he.shape[0], 3,
         he.shape[-2] // model.cfg.region, he.shape[-1] // model.cfg.region),
        generator=generator, device=device)
    prepared = reward_reference(reference, model.cfg, marker)
    rows = []

    with torch.inference_mode():
      for mode in ("fp32", "bf16"):
        with autocast_context(device, mode):
            times = grid(model, noise)
            state, base, observation = policy_state(model, policy, he, marker, noise, times)
            mean = policy(observation)
            actions = mean.unsqueeze(0) + model.cfg.action_std * action_epsilon
            no_action_raw = integrate(model, he, marker, noise, times)
            no_action = no_action_raw.clamp(-1, 1)
        reference_dab = rgb_to_dab(reference, model.cfg)
        tau = model.cfg.markers["HER2"]
        reference_support = (reference_dab > tau).float().mean().item()
        reference_dab_mean = reference_dab.mean().item()
        reward_values = []
        for candidate_index, action in enumerate(actions):
            with autocast_context(device, mode):
                corrected = corrected_velocity(model, base, action)
                delta_t = times[model.cfg.action_step + 1] - times[model.cfg.action_step]
                action_start = state + delta_t * corrected
                zero_action_start = state + delta_t * base
                action_raw = integrate(model, he, marker, action_start,
                                       times[model.cfg.action_step + 1:])
                generated = action_raw.clamp(-1, 1)
            # Always recompute reward and stain diagnostics in FP32, including
            # candidates whose image trajectories used BF16 autocast.
            components = paired_expression_reward_components(
                model, generated.float(), reference, marker, prepared=prepared)
            residual = corrected - base
            delta = generated.float() - no_action.float()
            raw_delta = action_raw.float() - no_action_raw.float()
            endpoint_delta = action_start.float() - zero_action_start.float()
            generated_dab = rgb_to_dab(generated.float(), model.cfg)
            reward = components["reward"].item()
            reward_values.append(reward)
            rows.append({
                    "pair_id": pair_id,
                    "source_prefix": pair_id.split("/", 1)[0],
                    "crop_seed": crop_seed,
                    "candidate": candidate_index,
                    "precision": mode,
                    "policy_mean_rms": mean.float().square().mean().sqrt().item(),
                    "action_std": model.cfg.action_std,
                    "action_rms": action.float().square().mean().sqrt().item(),
                    "action_abs_max": action.float().abs().max().item(),
                    "base_velocity_rms": base.float().square().mean().sqrt().item(),
                    "velocity_correction_rms": residual.float().square().mean().sqrt().item(),
                    "correction_to_base_rms": (residual.float().square().mean().sqrt()
                                                / base.float().square().mean().sqrt().clamp_min(1e-12)).item(),
                    "action_endpoint_delta_rms": endpoint_delta.square().mean().sqrt().item(),
                    "output_delta_mae_rgb01": delta.abs().mean().item() / 2,
                    "output_delta_rmse_rgb01": delta.square().mean().sqrt().item() / 2,
                    "output_delta_raw_mae_rgb01": raw_delta.abs().mean().item() / 2,
                    "reference_clip_fraction": ((no_action_raw < -1) | (no_action_raw > 1)).float().mean().item(),
                    "action_clip_fraction": ((action_raw < -1) | (action_raw > 1)).float().mean().item(),
                    "reward_total": reward,
                    "reward_support": components["support_score"].item(),
                    "reward_amount": components["amount_score"].item(),
                    "reference_dab_mean": reference_dab_mean,
                    "reference_dab_support_fraction": reference_support,
                    "output_dab_mean": generated_dab.mean().item(),
                    "output_dab_support_fraction": (generated_dab > tau).float().mean().item(),
                    "fp32_group_spread": None,
                    "fp32_signal": None,
                    "bf16_group_spread": None,
                    "bf16_signal": None,
            })
        groups = reward_values
        for row in rows[-candidates:]:
            row[f"{mode}_group_spread"] = max(groups) - min(groups)
            row[f"{mode}_signal"] = int(max(groups) - min(groups) >= model.cfg.min_reward_spread)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True, help="Flow-stage checkpoint or raw model state dict")
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261129)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    actual_hash = sha256(args.weights)
    if actual_hash.lower() != args.weights_sha256.lower():
        raise ValueError("Flow weights SHA-256 does not match the supplied provenance value")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing diagnostic results: {args.output}")
    cfg = read_config(args.config)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("Candidate diagnostic requires the configured CUDA device")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = StainFlow(cfg).to(device).eval()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    weights = checkpoint.get("model", checkpoint)
    if "model" in checkpoint:
        if checkpoint.get("stage") != "flow" or checkpoint.get("step") != args.checkpoint_step:
            raise ValueError("Checkpoint stage/step does not match the supplied Flow provenance")
        if checkpoint.get("config") != model.cfg.__dict__:
            raise ValueError("Supplied config differs from the Flow checkpoint config")
    model.load_state_dict(weights, strict=True)
    policy = VelocityPolicy(cfg).to(device).eval()
    dataset = IHCCrops(args.cache, cfg)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    selected = sample_states(dataset, split["pilot_dev_pair_ids"], args.states)
    if len({pair_id for pair_id, _ in selected}) != args.states:
        raise RuntimeError("Diagnostic states are not unique")

    started = time.monotonic()
    rows = []
    for index, (pair_id, dataset_index) in enumerate(selected, start=1):
        rows.extend(diagnose_one(model, policy, dataset, pair_id, dataset_index,
                                 device=device, candidates=args.candidates,
                                 seed=args.seed + index))
        if index % 4 == 0 or index == len(selected):
            torch.cuda.synchronize()
            print(json.dumps({"states": index, "total": len(selected),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)

    args.output.mkdir(parents=True)
    csv_path = args.output / "candidate_actions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = {}
    for mode in ("fp32", "bf16"):
        mode_rows = [row for row in rows if row["precision"] == mode]
        by_state = {}
        for row in mode_rows:
            by_state.setdefault(row["pair_id"], []).append(row)
        spreads = [group[0][f"{mode}_group_spread"] for group in by_state.values()]
        signals = [group[0][f"{mode}_signal"] for group in by_state.values()]
        for field in ("action_rms", "velocity_correction_rms", "correction_to_base_rms",
                      "action_endpoint_delta_rms", "output_delta_mae_rgb01",
                      "output_delta_raw_mae_rgb01", "reference_clip_fraction", "action_clip_fraction",
                      "reward_total", "reward_support", "reward_amount",
                      "output_dab_mean", "output_dab_support_fraction"):
            values = [row[field] for row in mode_rows]
            summaries.setdefault(mode, {})[field] = {
                "mean": statistics.fmean(values), "median": statistics.median(values)}
        summaries[mode]["group_reward_spread"] = {
            "mean": statistics.fmean(spreads), "median": statistics.median(spreads),
            "min": min(spreads), "max": max(spreads),
        }
        summaries[mode]["signal_fraction_at_existing_gate"] = statistics.fmean(signals)

    report = {
        "diagnostic_only_no_weight_updates": True,
        "source_commit": args.source_commit,
        "flow_checkpoint_step": args.checkpoint_step,
        "weights_sha256": actual_hash,
        "config_sha256": sha256(args.config),
        "pilot_dev_split_sha256": sha256(args.split),
        "pilot_dev_cache_index_sha256": dataset.fingerprint,
        "state_count": len(selected),
        "source_prefix_count_in_diagnostic": len({pair_id.split("/", 1)[0] for pair_id, _ in selected}),
        "candidate_count_per_state": args.candidates,
        "action_sampling": "policy mean + action_std * fixed standard-normal epsilon",
        "reward_recomputation": "FP32 for candidates from both FP32 and BF16 trajectories",
        "action_std": cfg.action_std,
        "reward_signal_threshold_unchanged": cfg.min_reward_spread,
        "precisions": summaries,
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device)},
        "csv_sha256": sha256(csv_path),
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
