"""Shared input and checkpoint rules for the two MIST HER2 evaluators."""

import csv
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from morphoflow_ihc.config import Config
from morphoflow_ihc.data import verify_manifest_image
from morphoflow_ihc.engine import CHECKPOINT_FORMAT
from morphoflow_ihc.model import StainFlow, VelocityPolicy
from morphoflow_ihc.trajectory import sample_rgb


def read_pairs(manifest, count):
    pairs = {}
    training = set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            split = row["split"].strip()
            key = tuple(row[field].strip() for field in ("source_prefix", "filename"))
            if any(not part or part in {".", ".."} or any(c in part for c in "/\\:,") for part in key):
                raise ValueError(f"Invalid manifest path for {key}")
            if split in {"trainA", "trainB"}:
                training.add(key)
                continue
            if split not in {"valA", "valB"}:
                raise ValueError(f"Unexpected MIST split: {split}")
            pair = pairs.setdefault(key, {})
            if split in pair:
                raise ValueError(f"Duplicate {split} row for {key}")
            pair[split] = row
    if training & pairs.keys():
        raise ValueError("A pair appears in both training and evaluation")
    if len(pairs) != count or any(set(pair) != {"valA", "valB"} for pair in pairs.values()):
        raise ValueError(f"Expected {count} complete official pairs, found {len(pairs)} pair keys")
    if len({Path(key[1]).stem for key in pairs}) != count:
        raise ValueError("Official pairs contain colliding filename stems")
    return sorted(pairs.items())


def official_pairs(manifest):
    canonical = Path(__file__).resolve().parents[2] / "datasets" / "MIST_HER2" / "manifest.csv"
    pairs = read_pairs(manifest, 1000)
    if manifest.resolve() != canonical.resolve():
        def cohort_signature(cohort):
            return [
                (key, tuple((int(pair[split]["bytes"]), pair[split]["sha256"].strip().lower())
                            for split in ("valA", "valB")))
                for key, pair in cohort
            ]

        if cohort_signature(pairs) != cohort_signature(read_pairs(canonical, 1000)):
            raise ValueError("Evaluation rows differ from the repository's official MIST HER2 cohort")
    return pairs


def image_seed(base_seed, pair_key):
    token = f"{int(base_seed)}:{pair_key[0]}/{pair_key[1]}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "big") & ((1 << 63) - 1)


def read_pair(data_root, key, pair, cfg):
    paths = [data_root / split / key[0] / key[1] for split in ("valA", "valB")]
    images = []
    for split, path in zip(("valA", "valB"), paths):
        row = pair[split]
        verify_manifest_image(path, row)
        if float(row.get("mpp") or cfg.pixel_size_um) != cfg.pixel_size_um:
            raise ValueError(f"Native MIST evaluation requires canonical pixel spacing: {path}")
        with Image.open(path) as raw:
            if raw.size != (1024, 1024):
                raise ValueError(f"Expected a native 1024x1024 MIST image: {path}")
            images.append(np.asarray(raw.convert("RGB")))
    return paths, images


def checkpoint_for_evaluation(args, cfg, device, *, load_model=True):
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    full = isinstance(state, dict) and "model" in state
    checkpoint_format = state.get("format") if full else None
    stage = state.get("stage") if full else "flow"
    step = state.get("step") if full else args.checkpoint_step
    if checkpoint_format != CHECKPOINT_FORMAT and (
        not args.allow_historical_weights or checkpoint_format not in {None, 9}
    ):
        raise ValueError("Use a format 10 checkpoint, or explicitly allow historical/bare weights for evaluation")
    if stage not in {"flow", "rl"} or type(step) is not int or step < 1:
        raise ValueError("Expected a Flow or RL checkpoint with a positive training step")
    if args.checkpoint_step is not None and step != args.checkpoint_step:
        raise ValueError("--checkpoint-step disagrees with embedded metadata")
    if full:
        embedded = dict(state["config"])
        embedded.pop("advantage_scale", None)
        if stage == "rl" and checkpoint_format == 9:
            embedded.setdefault("trajectory_candidates", 2)
        embedded = Config(**embedded)
        # The joint CLI can override only this training rate.
        if stage == "rl":
            embedded.rl_lr = cfg.rl_lr
        if embedded != cfg:
            raise ValueError("Evaluation config does not match the checkpoint configuration")
    if args.disable_policy and stage != "rl":
        raise ValueError("--disable-policy requires an RL checkpoint")
    use_policy = stage == "rl" and not args.disable_policy
    meta = {"checkpoint_stage": stage, "checkpoint_step": step,
            "checkpoint_format": checkpoint_format,
            "training_objective_verified": checkpoint_format == CHECKPOINT_FORMAT,
            "policy_enabled": use_policy,
            "policy_objective": state.get("policy_objective", "ppo") if stage == "rl" else None}
    model = policy = None
    if load_model:
        model = StainFlow(cfg)
        model.load_state_dict(state["model"] if full else state, strict=True)
        model.to(device).eval()
        if use_policy:
            policy = VelocityPolicy(cfg)
            policy.load_state_dict(state["policy"], strict=True)
            policy.to(device).eval()
    return model, policy, meta


@torch.inference_mode()
def predict_rgb(model, policy, rgb, marker, seed):
    factor = 2 ** (len(model.cfg.channel_mult) - 1)
    if any(side % factor for side in rgb.shape[:2]):
        raise ValueError(f"Native image geometry is incompatible with the U-Net: {rgb.shape}")
    he = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(127.5).sub(1)[None].to(marker.device)
    noise = torch.randn(he.shape[1:], generator=torch.Generator().manual_seed(seed))[None].to(marker.device)
    rendered = sample_rgb(model, he, marker, noise, policy)
    if not torch.isfinite(rendered).all():
        raise FloatingPointError("Nonfinite model output")
    return np.rint((rendered.cpu()[0].permute(1, 2, 0).numpy() + 1) * 127.5).astype(np.uint8)
