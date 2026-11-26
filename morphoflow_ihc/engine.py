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
from .model import StainFlow, VelocityPolicy
from .stains import canonical_size, decompose
from .trajectory import collect, direct_policy_update, policy_update, sample_rgb, terminal_reward


def choose_device(name):
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch build")
    return device


def precision(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def save_checkpoint(path, model, optimizer, step, stage, policy=None, critic_optimizer=None, policy_objective=None,
                    flow_optimizer=None):
    state = {
        "format": 8, "stage": stage, "step": step, "config": asdict(model.cfg),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
    }
    if critic_optimizer is not None:
        state["critic_optimizer"] = critic_optimizer.state_dict()
    if flow_optimizer is not None:
        state["flow_optimizer"] = flow_optimizer.state_dict()
    if policy is not None:
        state["policy"] = policy.state_dict()
        state["policy_objective"] = policy_objective
    atomic_write(path, lambda stream: torch.save(state, stream))


def open_checkpoint(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format") != 8 or state.get("stage") not in {"flow", "rl"}:
        raise ValueError("This checkpoint predates paired PatchNCE/PatchGAN training. Train a new format 8 checkpoint")
    model = StainFlow(Config(**state["config"])).to(device)
    model.load_state_dict(state["model"])
    policy = None
    if state["stage"] == "rl":
        policy = VelocityPolicy(model.cfg).to(device)
        if state.get("policy", {}).get("net.4.weight", torch.empty(0)).shape[0] != 3:
            raise ValueError("Incompatible policy checkpoint: retrain the RGB-channel policy")
        policy.load_state_dict(state["policy"])
    return model, policy, state


def batches(loader):
    while True:
        yield from loader


def make_loader(dataset, batch, workers, device):
    return DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=workers,
                      pin_memory=device.type == "cuda", persistent_workers=workers > 0)


def flow_optimizers(model):
    parameters = [param for name, param in model.named_parameters() if not name.startswith("critic.")]
    return (torch.optim.AdamW(parameters, lr=model.cfg.flow_lr),
            torch.optim.AdamW(model.critic.parameters(), lr=model.cfg.critic_lr))


def flow_update(model, optimizer, critic_optimizer, image, he, marker, policy=None):
    cfg = model.cfg
    optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    policy_grad_state = [p.requires_grad for p in policy.parameters()] if policy is not None else None
    if policy is not None:
        # The policy is optimized by PPO/direct updates; here its action is a
        # fixed controller, while gradients still pass through its inputs to G.
        for parameter in policy.parameters():
            parameter.grad = None
        policy.requires_grad_(False)
    try:
        with precision(he.device):
            flow_loss, nce_loss, fake = model.losses(image, he, marker, policy=policy)
            critic_loss = 0.5 * (F.softplus(-model.critic(image, he, marker)).mean()
                                 + F.softplus(model.critic(fake.detach(), he, marker)).mean())
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Nonfinite critic loss")
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0, error_if_nonfinite=True)
        critic_optimizer.step()
        critic_optimizer.zero_grad(set_to_none=True)

        model.critic.requires_grad_(False)
        try:
            with precision(he.device):
                adversarial_loss = F.softplus(-model.critic(fake, he, marker)).mean()
                loss = flow_loss + cfg.nce_weight * nce_loss + cfg.adversarial_weight * adversarial_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite flow training loss")
            loss.backward()
            parameters = [param for group in optimizer.param_groups for param in group["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
        finally:
            model.critic.requires_grad_(True)
    finally:
        if policy is not None:
            for parameter, requires_grad in zip(policy.parameters(), policy_grad_state):
                parameter.requires_grad_(requires_grad)
    stats = {"flow_loss": flow_loss.item(), "nce_loss": nce_loss.item(),
             "critic_loss": critic_loss.item(), "adversarial_loss": adversarial_loss.item()}
    return stats


def train_flow(args, cfg):
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    model = StainFlow(cfg).to(device)
    step = 0
    if args.resume:
        model, _, state = open_checkpoint(args.resume, device)
        if state["stage"] != "flow" or model.cfg != cfg:
            raise ValueError("Resume requires a flow checkpoint with the same model configuration")
        step = state["step"]
    optimizer, critic_optimizer = flow_optimizers(model)
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    dataset = IHCCrops(args.cache, cfg)
    loader = make_loader(dataset, args.batch, args.workers, device)
    target = args.steps or args.epochs * len(loader)
    deadline = time.monotonic() + 3600 * args.hours
    for pack in batches(loader):
        if step >= target or time.monotonic() >= deadline:
            break
        marker, image, he = (value.to(device, non_blocking=True) for value in pack)
        stats = flow_update(model, optimizer, critic_optimizer, image, he, marker)
        step += 1
        if step == 1 or step % args.save_every == 0 or step == target:
            save_checkpoint(args.output, model, optimizer, step, "flow", critic_optimizer=critic_optimizer)
            print(json.dumps({"stage": "flow", "step": step, **stats}), flush=True)
        if step >= target or time.monotonic() >= deadline:
            break
    save_checkpoint(args.output, model, optimizer, step, "flow", critic_optimizer=critic_optimizer)
    print(json.dumps({"stage": "flow", "step": step, "target": target, "file": str(args.output)}))


def train_policy(args):
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    model, policy, state = open_checkpoint(args.checkpoint, device)
    cfg = model.cfg
    if args.advantage_scale is not None:
        cfg.advantage_scale = args.advantage_scale
    if args.policy_lr is not None:
        cfg.rl_lr = args.policy_lr
    flow_optimizer, critic_optimizer = flow_optimizers(model)
    if state["stage"] == "rl" and "flow_optimizer" not in state:
        raise ValueError("Start joint training from a base flow checkpoint or resume a joint checkpoint")
    flow_optimizer.load_state_dict(state["flow_optimizer"] if state["stage"] == "rl" else state["optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])
    if policy is None:
        policy = VelocityPolicy(cfg).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.rl_lr)
    groups = 0
    objective = args.objective
    if state["stage"] == "rl":
        if (state.get("policy_objective") or "ppo") != objective:
            raise ValueError("Start both objectives from the same base flow checkpoint")
        groups = state["step"]
        optimizer.load_state_dict(state["optimizer"])
        if args.policy_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = cfg.rl_lr
    dataset = IHCCrops(args.cache, cfg)
    loader = make_loader(dataset, args.batch, args.workers, device)
    next_save = groups + args.save_every
    deadline = time.monotonic() + 3600 * args.hours
    for pack in batches(loader):
        if groups >= args.groups or time.monotonic() >= deadline:
            break
        limit = args.groups - groups
        marker, image, he = (value[:limit].to(device, non_blocking=True) for value in pack)
        direct = objective == "direct"
        # Keep this rollout's environment fixed through its policy update.
        model.eval().requires_grad_(False)
        try:
            with precision(device), torch.set_grad_enabled(direct):
                experience = collect(model, policy, he, image, marker, direct=direct)
            update = direct_policy_update if direct else policy_update
            stats = update(policy, optimizer, experience, cfg)
        finally:
            model.train().requires_grad_(True)
        stats.update(flow_update(model, flow_optimizer, critic_optimizer, image, he, marker, policy=policy))
        stats["objective"] = objective
        groups += he.shape[0]
        if groups >= next_save or groups == args.groups or groups == he.shape[0]:
            save_checkpoint(args.output, model, optimizer, groups, "rl", policy, critic_optimizer=critic_optimizer,
                            policy_objective=objective, flow_optimizer=flow_optimizer)
            next_save = groups + args.save_every
            print(json.dumps({"stage": "rl", "groups": groups, **stats}), flush=True)
        if groups >= args.groups or time.monotonic() >= deadline:
            break
    save_checkpoint(args.output, model, optimizer, groups, "rl", policy, critic_optimizer=critic_optimizer,
                            policy_objective=objective, flow_optimizer=flow_optimizer)
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
    native_size = source.size
    size = canonical_size(native_size, args.mpp if args.mpp is not None else cfg.pixel_size_um, cfg)
    scaled = source.resize(size, Image.Resampling.BILINEAR) if size != native_size else source
    rgb = np.asarray(scaled)
    height, width = rgb.shape[:2]
    tile = cfg.crop
    padded_shape = max(height, tile), max(width, tile)
    image = np.pad(rgb, ((0,padded_shape[0]-height),(0,padded_shape[1]-width),(0,0)), constant_values=255)
    image = torch.from_numpy(image.copy()).permute(2,0,1).float() / 127.5 - 1
    noise = torch.randn((3,*padded_shape), generator=torch.Generator().manual_seed(args.seed))
    total = torch.zeros((3,*padded_shape))
    weight_total = torch.zeros(padded_shape)
    taper = torch.hann_window(tile, periodic=False).clamp_min(0.05)
    blend = taper[:,None] * taper[None,:]
    for y in offsets(height, tile, tile-cfg.overlap):
        for x in offsets(width, tile, tile-cfg.overlap):
            region = (slice(y,y+tile),slice(x,x+tile))
            image_tile = image[:,region[0],region[1]][None].to(device)
            noise_tile = noise[:,region[0],region[1]][None].to(device)
            with precision(device):
                estimate = sample_rgb(model, image_tile, marker, noise_tile, policy)
            total[:,region[0],region[1]] += estimate.float().cpu()[0] * blend
            weight_total[region] += blend
    result = (total / weight_total)[:,:height,:width].permute(1,2,0).clamp(-1,1).numpy()
    rendered = Image.fromarray(np.rint((result + 1)*127.5).astype(np.uint8)).resize(native_size, Image.Resampling.BILINEAR)
    atomic_write(args.output, lambda stream: rendered.save(stream, format="PNG"))
    if args.maps:
        _, dab, _ = decompose(np.asarray(rendered), cfg, "IHC")
        atomic_write(args.maps, lambda stream: np.savez_compressed(stream, dab=dab))
    print(json.dumps({"marker":args.marker.upper(), "width":source.width, "height":source.height,
                      "rl":policy is not None, "file":str(args.output)}))


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
