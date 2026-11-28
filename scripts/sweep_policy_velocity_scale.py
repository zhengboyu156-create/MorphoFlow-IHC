"""No-update sensitivity sweep of PPO candidate actions over velocity limits."""

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from morphoflow_ihc.data import IHCCrops
from morphoflow_ihc.engine import CHECKPOINT_FORMAT, open_checkpoint
from morphoflow_ihc.trajectory import (
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--states", type=int, default=32)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--velocity-limits", default="0.1,0.25,0.5,1.0")
    parser.add_argument("--seed", type=int, default=20261130)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing action-scale sweep: {args.output}")
    limits = [float(value) for value in args.velocity_limits.split(",")]
    if not limits or any(value <= 0 for value in limits):
        raise ValueError("Velocity limits must be positive comma-separated values")
    device = torch.device(args.device)
    model, policy, state = open_checkpoint(args.checkpoint, device)
    if state.get("format") != CHECKPOINT_FORMAT or state.get("stage") != "rl" or policy is None:
        raise ValueError("Expected a current-format joint PPO checkpoint")
    cfg = model.cfg
    dataset = IHCCrops(args.cache, cfg)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    index_by_id = {entry["pair_id"]: index for index, entry in enumerate(dataset.entries)}
    by_prefix = {}
    for pair_id in sorted(split["pilot_dev_pair_ids"]):
        by_prefix.setdefault(pair_id.split("/", 1)[0], pair_id)
    selected = [by_prefix[prefix] for prefix in sorted(by_prefix)][:args.states]
    if len(selected) != args.states:
        raise ValueError("Pilot-dev does not contain the requested unique source-prefix states")

    rows = []
    started = time.monotonic()
    for state_index, pair_id in enumerate(selected):
        crop_seed = int.from_bytes(hashlib.sha256(f"{args.seed}:{pair_id}".encode()).digest()[:8], "big") % (2**63 - 1)
        marker_value, reference, he = dataset[(index_by_id[pair_id], crop_seed)]
        marker = torch.tensor([marker_value], device=device)
        reference, he = reference[None].to(device), he[None].to(device)
        torch.manual_seed(crop_seed)
        torch.cuda.manual_seed_all(crop_seed)
        noise = torch.randn_like(he)
        epsilon_seed = (crop_seed + 1709) % (2**63 - 1)
        action_generator = torch.Generator(device=device).manual_seed(epsilon_seed)
        times = grid(model, noise)
        with torch.inference_mode():
            flow_state, base, observation = policy_state(model, policy, he, marker, noise, times)
            mean = policy(observation)
            epsilon = torch.randn((args.candidates, *mean.shape), generator=action_generator, device=device)
            actions = mean.unsqueeze(0) + cfg.action_std * epsilon
            no_action = integrate(model, he, marker, noise, times).clamp(-1, 1)
            prepared = reward_reference(reference, cfg, marker)
            reward_ref_dab = rgb_to_dab(reference, cfg)
            for limit in limits:
                cfg.velocity_limit = limit
                candidate_rows = []
                for candidate_index, action in enumerate(actions):
                    corrected = corrected_velocity(model, base, action)
                    dt = times[cfg.action_step + 1] - times[cfg.action_step]
                    initial = flow_state + dt * corrected
                    raw = integrate(model, he, marker, initial, times[cfg.action_step + 1:])
                    generated = raw.clamp(-1, 1)
                    components = paired_expression_reward_components(model, generated, reference, marker, prepared=prepared)
                    residual = corrected - base
                    row = {
                        "pair_id": pair_id,
                        "source_prefix": pair_id.split("/", 1)[0],
                        "candidate": candidate_index,
                        "velocity_limit": limit,
                        "policy_mean_rms": mean.square().mean().sqrt().item(),
                        "sampled_action_rms": action.square().mean().sqrt().item(),
                        "velocity_correction_rms": residual.square().mean().sqrt().item(),
                        "correction_to_base_rms": (residual.square().mean().sqrt()
                                                    / base.square().mean().sqrt().clamp_min(1e-12)).item(),
                        "output_delta_mae_rgb01": (generated - no_action).abs().mean().item() / 2,
                        "output_clip_fraction": ((raw < -1) | (raw > 1)).float().mean().item(),
                        "reward_total": components["reward"].item(),
                        "reward_support": components["support_score"].item(),
                        "reward_amount": components["amount_score"].item(),
                        "reference_dab_mean": reward_ref_dab.mean().item(),
                        "output_dab_mean": rgb_to_dab(generated, cfg).mean().item(),
                    }
                    candidate_rows.append(row)
                spread = max(row["reward_total"] for row in candidate_rows) - min(row["reward_total"] for row in candidate_rows)
                for row in candidate_rows:
                    row["group_reward_spread"] = spread
                    row["signal_at_unchanged_gate"] = int(spread >= cfg.min_reward_spread)
                    rows.append(row)
        if (state_index + 1) % 8 == 0 or state_index + 1 == len(selected):
            torch.cuda.synchronize()
            print(json.dumps({"states": state_index + 1, "total": len(selected),
                              "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)

    args.output.mkdir(parents=True)
    csv_path = args.output / "velocity_scale_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "no_weight_updates": True,
        "checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_step": state["step"],
        "pilot_dev_split_sha256": sha256(args.split),
        "pilot_dev_cache_index_sha256": dataset.fingerprint,
        "state_count": len(selected),
        "source_prefix_count": len(set(row["source_prefix"] for row in rows)),
        "candidate_count": args.candidates,
        "velocity_limits": limits,
        "reward_gate_unchanged": cfg.min_reward_spread,
        "action_std": cfg.action_std,
        "results": {},
        "csv_sha256": sha256(csv_path),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "runtime": {"torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device)},
    }
    for limit in limits:
        subset = [row for row in rows if row["velocity_limit"] == limit]
        spreads = [row["group_reward_spread"] for row in subset[::args.candidates]]
        summary["results"][str(limit)] = {
            metric: {"mean": statistics.fmean(row[metric] for row in subset),
                     "median": statistics.median(row[metric] for row in subset)}
            for metric in ("policy_mean_rms", "sampled_action_rms", "velocity_correction_rms",
                           "correction_to_base_rms", "output_delta_mae_rgb01", "output_clip_fraction",
                           "reward_total", "reward_support", "reward_amount", "reference_dab_mean", "output_dab_mean")
        }
        summary["results"][str(limit)]["group_reward_spread"] = {
            "mean": statistics.fmean(spreads), "median": statistics.median(spreads),
            "min": min(spreads), "max": max(spreads),
        }
        summary["results"][str(limit)]["signal_fraction_at_unchanged_gate"] = statistics.fmean(
            row["signal_at_unchanged_gate"] for row in subset[::args.candidates])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
