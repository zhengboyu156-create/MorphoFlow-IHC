"""Training, checkpointing and native-size virtual IHC inference."""

from contextlib import nullcontext
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import IHCCrops, atomic_write
from .model import StainFlow, VelocityPolicy, broadcast_grid
from .stains import canonical_size, decompose, render_rgb, resize_map
from .trajectory import collect, policy_update, sample_dab, terminal_reward


def choose_device(name):
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch build")
    return device


def precision(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def save_checkpoint(path, model, optimizer, step, stage, policy=None):
    state = {
        "format": 1, "stage": stage, "step": step, "config": asdict(model.cfg),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
    }
    if policy is not None:
        state["policy"] = policy.state_dict()
    atomic_write(path, lambda stream: torch.save(state, stream))


def open_checkpoint(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format") != 1 or state.get("stage") not in {"flow", "rl"}:
        raise ValueError("Unrecognized checkpoint format")
    model = StainFlow(Config(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    policy = None
    if state["stage"] == "rl":
        policy = VelocityPolicy(model.cfg).to(device)
        policy.load_state_dict(state["policy"])
    return model, policy, state


def batches(loader):
    while True:
        yield from loader


def make_loader(dataset, batch, workers, device):
    return DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers,
                      pin_memory=device.type == "cuda", persistent_workers=workers > 0)


def train_flow(args, cfg):
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    model = StainFlow(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.flow_lr)
    step = 0
    if args.resume:
        model, _, state = open_checkpoint(args.resume, device)
        if state["stage"] != "flow" or model.cfg != cfg:
            raise ValueError("Resume requires a flow checkpoint with the same model configuration")
        step = state["step"]
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.flow_lr)
        optimizer.load_state_dict(state["optimizer"])
    dataset = IHCCrops(args.cache, cfg)
    loader = make_loader(dataset, args.batch, args.workers, device)
    target = args.steps or args.epochs * len(loader)
    deadline = time.monotonic() + 3600 * args.hours
    for pack in batches(loader):
        if step >= target or time.monotonic() >= deadline:
            break
        h, reference, tissue, marker = (value.to(device, non_blocking=True) for value in pack)
        optimizer.zero_grad(set_to_none=True)
        with precision(device):
            region_loss, flow_loss = model.losses(h / cfg.h_scale, reference, tissue, marker)
            loss = region_loss + flow_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite supervised loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step == 1 or step % args.save_every == 0 or step == target:
            save_checkpoint(args.output, model, optimizer, step, "flow")
            print(json.dumps({"stage": "flow", "step": step, "region_loss": region_loss.item(),
                              "flow_loss": flow_loss.item()}), flush=True)
        if step >= target or time.monotonic() >= deadline:
            break
    save_checkpoint(args.output, model, optimizer, step, "flow")
    print(json.dumps({"stage": "flow", "step": step, "target": target, "file": str(args.output)}))


def train_policy(args):
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    model, policy, state = open_checkpoint(args.checkpoint, device)
    cfg = model.cfg
    model.eval().requires_grad_(False)
    if policy is None:
        policy = VelocityPolicy(cfg).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.rl_lr)
    groups = 0
    if state["stage"] == "rl":
        groups = state["step"]
        optimizer.load_state_dict(state["optimizer"])
    dataset = IHCCrops(args.cache, cfg, prioritize_positive=True)
    loader = make_loader(dataset, args.batch, args.workers, device)
    next_save = groups + args.save_every
    deadline = time.monotonic() + 3600 * args.hours
    for pack in batches(loader):
        if groups >= args.groups or time.monotonic() >= deadline:
            break
        limit = args.groups - groups
        h, reference, tissue, marker = (value[:limit].to(device, non_blocking=True) for value in pack)
        with precision(device):
            experience = collect(model, policy, h / cfg.h_scale, reference, tissue, marker)
        stats = policy_update(policy, optimizer, experience, cfg)
        groups += h.shape[0]
        if groups >= next_save or groups == args.groups or groups == h.shape[0]:
            save_checkpoint(args.output, model, optimizer, groups, "rl", policy)
            next_save = groups + args.save_every
            print(json.dumps({"stage": "rl", "groups": groups, **stats}), flush=True)
        if groups >= args.groups or time.monotonic() >= deadline:
            break
    save_checkpoint(args.output, model, optimizer, groups, "rl", policy)
    print(json.dumps({"stage": "rl", "groups": groups, "target": args.groups, "file": str(args.output)}))


def offsets(length, tile, stride):
    last = max(0, length - tile)
    result = list(range(0, last + 1, stride))
    if result[-1] != last:
        result.append(last)
    return result


@torch.no_grad()
def infer(args):
    device = choose_device(args.device)
    model, policy, _ = open_checkpoint(args.checkpoint, device)
    model.eval()
    if policy is not None:
        policy.eval()
    cfg = model.cfg
    marker = torch.tensor([cfg.marker_id(args.marker)], device=device)
    with Image.open(args.image) as raw:
        source = raw.convert("RGB")
    native_h, _, native_tissue = decompose(np.asarray(source), cfg, "HE")
    size = canonical_size(source.size, args.mpp if args.mpp is not None else cfg.pixel_size_um, cfg)
    scaled_h = resize_map(native_h, size)
    scaled_tissue = resize_map(native_tissue.astype(np.float32), size) >= 0.5
    height, width = scaled_h.shape
    tile = cfg.crop
    padded_shape = max(height, tile), max(width, tile)
    pad = ((0, padded_shape[0] - height), (0, padded_shape[1] - width))
    h = torch.from_numpy(np.pad(scaled_h / cfg.h_scale, pad))
    tissue = torch.from_numpy(np.pad(scaled_tissue, pad))
    noise_gen = torch.Generator().manual_seed(args.seed)
    noise = torch.randn(padded_shape, generator=noise_gen)
    dab_total = torch.zeros(padded_shape)
    prob_total = torch.zeros_like(dab_total)
    weight_total = torch.zeros_like(dab_total)
    taper = torch.hann_window(tile, periodic=False).clamp_min(0.05)
    blend = taper[:, None] * taper[None, :]
    stride = tile - cfg.overlap
    for y in offsets(height, tile, stride):
        for x in offsets(width, tile, stride):
            region = (slice(y, y + tile), slice(x, x + tile))
            h_tile, tissue_tile, noise_tile = (a[region][None, None].to(device) for a in (h, tissue, noise))
            with precision(device):
                estimate, probability = sample_dab(model, h_tile, tissue_tile, marker, noise_tile, policy)
            dab_total[region] += estimate.float().cpu()[0, 0] * blend
            prob_total[region] += broadcast_grid(probability, h_tile).float().cpu()[0, 0] * blend
            weight_total[region] += blend
    dab = resize_map((dab_total / weight_total)[:height, :width].numpy(), source.size)
    probability = resize_map((prob_total / weight_total)[:height, :width].numpy(), source.size)
    if not np.isfinite(dab).all():
        raise FloatingPointError("Inference produced nonfinite DAB")
    corrected_h = cfg.counterstain_gain * native_h
    rendered = render_rgb(corrected_h, dab, cfg)
    # PNG makes the supplied coordinate grid and rendered RGB reproducible.
    atomic_write(args.output, lambda stream: rendered.save(stream, format="PNG"))
    if args.maps:
        atomic_write(args.maps, lambda stream: np.savez_compressed(
            stream, h=corrected_h, dab=dab, probability=probability
        ))
    print(json.dumps({"marker": args.marker.upper(), "width": source.width, "height": source.height,
                      "rl": policy is not None, "file": str(args.output)}))


def evaluate(args, cfg):
    with np.load(args.prediction, allow_pickle=False) as maps:
        estimated = maps["dab"]
    with Image.open(args.reference) as raw:
        _, true_dab, tissue = decompose(np.asarray(raw.convert("RGB")), cfg, "IHC")
    if estimated.shape != true_dab.shape or estimated.ndim != 2 or not np.isfinite(estimated).all():
        raise ValueError("The saved DAB and paired IHC must share coordinates, shape and finite values")
    tau = list(cfg.markers.values())[cfg.marker_id(args.marker)]
    region = cfg.region
    height, width = estimated.shape
    pad = (0, -width % region, 0, -height % region)
    estimate, actual = (F.pad(torch.from_numpy(a.astype(np.float32))[None, None], pad) for a in (estimated, true_dab))
    seen = F.pad(torch.from_numpy(tissue)[None, None], pad)
    if not seen.any():
        raise ValueError("Reference contains no evaluable tissue")
    count = F.avg_pool2d(seen.float(), region)
    phases = [F.avg_pool2d(((a > tau) & seen).float(), region) / count.clamp_min(1e-8) for a in (estimate, actual)]
    metrics = {
        "marker": args.marker.upper(), "threshold": tau, "region_px": region,
        "region_support_fraction_mae": (phases[0] - phases[1]).abs()[count > 0].mean().item(),
        "reference_tissue_fraction": seen.float().sum().item() / (height * width),
    }
    if args.same_section:
        metrics["pixel_dab_mae"] = (estimate - actual).abs()[seen].mean().item()
        metrics["pixel_support_reward"] = terminal_reward(
            estimate, actual, seen, torch.tensor([[[[tau]]]]), region, cfg.intensity_weight
        ).item()
    report = json.dumps(metrics, indent=2, allow_nan=False)
    if args.output:
        atomic_write(args.output, lambda stream: stream.write(report.encode()))
    print(report)
