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


def save_checkpoint(path, model, optimizer, step, stage, policy=None, critic_optimizer=None, policy_objective=None):
    state = {
        "format": 3, "stage": stage, "step": step, "config": asdict(model.cfg),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
    }
    if model.expression_info is not None:
        state["expression_info"] = model.expression_info
    if critic_optimizer is not None:
        state["critic_optimizer"] = critic_optimizer.state_dict()
    if policy is not None:
        state["policy"] = policy.state_dict()
        state["policy_objective"] = policy_objective
    atomic_write(path, lambda stream: torch.save(state, stream))


def open_checkpoint(path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format") != 3 or state.get("stage") not in {"flow", "rl"}:
        raise ValueError("Unrecognized checkpoint format")
    model = StainFlow(Config(**state["config"])).to(device)
    if "expression_info" in state:
        model.attach_expression(state["expression_info"])
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


def training_dataset(args, model):
    info = model.expression_info
    if bool(info) != bool(args.expression_manifest):
        raise ValueError("Expression training requires a matching --expression-manifest and validated scorers")
    return IHCCrops(args.cache, model.cfg, expression_manifest=args.expression_manifest,
                    expected_digest=info["manifest_digest"] if info else None)


def train_flow(args, cfg):
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    model = StainFlow(cfg).to(device)
    step = 0
    if args.resume:
        if args.expression:
            raise ValueError("Resume uses the expression scorers stored in the checkpoint")
        model, _, state = open_checkpoint(args.resume, device)
        if state["stage"] != "flow" or model.cfg != cfg:
            raise ValueError("Resume requires a flow checkpoint with the same model configuration")
        step = state["step"]
    elif args.expression:
        model.attach_expression(torch.load(args.expression, map_location=device, weights_only=True))
    generator_params = [p for name, p in model.named_parameters()
                        if not name.startswith("critic.") and p.requires_grad]
    optimizer = torch.optim.AdamW(generator_params, lr=cfg.flow_lr)
    critic_optimizer = torch.optim.AdamW(model.critic.parameters(), lr=cfg.critic_lr)
    if args.resume:
        optimizer.load_state_dict(state["optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    dataset = training_dataset(args, model)
    loader = make_loader(dataset, args.batch, args.workers, device)
    target = args.steps or args.epochs * len(loader)
    deadline = time.monotonic() + 3600 * args.hours
    for pack in batches(loader):
        if step >= target or time.monotonic() >= deadline:
            break
        h, reference, tissue, marker, image, he, *labels = (value.to(device, non_blocking=True) for value in pack)
        optimizer.zero_grad(set_to_none=True)
        critic_marker = marker * 2 + labels[0].long() if labels else marker
        with precision(device):
            region_loss, flow_loss, nce_loss, fake = model.losses(h / cfg.h_scale, reference, tissue, marker, image, he)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss = (F.softplus(-model.critic(image, critic_marker)).mean() + F.softplus(model.critic(fake.detach(), critic_marker)).mean()) * 0.5
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Nonfinite critic loss")
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0, error_if_nonfinite=True)
        critic_optimizer.step()
        critic_optimizer.zero_grad(set_to_none=True)
        model.critic.requires_grad_(False)
        with precision(device):
            adversarial_loss = F.softplus(-model.critic(fake, critic_marker)).mean()
            expression_loss = fake.new_zeros(())
            if labels:
                expression_score = model.expression_scorers["IHC"](fake, marker)
                expression_loss = F.binary_cross_entropy_with_logits(expression_score, labels[0].float())
            loss = (region_loss + flow_loss + cfg.nce_weight * nce_loss
                    + cfg.adversarial_weight * adversarial_loss + cfg.expression_weight * expression_loss)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite supervised loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator_params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.critic.requires_grad_(True)
        step += 1
        if step == 1 or step % args.save_every == 0 or step == target:
            save_checkpoint(args.output, model, optimizer, step, "flow", critic_optimizer=critic_optimizer)
            print(json.dumps({"stage": "flow", "step": step, "region_loss": region_loss.item(),
                              "flow_loss": flow_loss.item(), "nce_loss": nce_loss.item(),
                              "critic_loss": critic_loss.item(), "adversarial_loss": adversarial_loss.item(),
                              "expression_loss": expression_loss.item()}), flush=True)
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
    model.eval().requires_grad_(False)
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
    dataset = training_dataset(args, model)
    loader = make_loader(dataset, args.batch, args.workers, device)
    next_save = groups + args.save_every
    deadline = time.monotonic() + 3600 * args.hours
    for pack in batches(loader):
        if groups >= args.groups or time.monotonic() >= deadline:
            break
        limit = args.groups - groups
        h, _, _, marker, _, he, *labels = (value[:limit].to(device, non_blocking=True) for value in pack)
        direct = objective == "direct"
        with precision(device), torch.set_grad_enabled(direct):
            experience = collect(model, policy, he, marker, labels[0].float() if labels else None, direct=direct)
        update = direct_policy_update if direct else policy_update
        stats = update(policy, optimizer, experience, cfg)
        stats["objective"] = objective
        groups += h.shape[0]
        if groups >= next_save or groups == args.groups or groups == h.shape[0]:
            save_checkpoint(args.output, model, optimizer, groups, "rl", policy, policy_objective=objective)
            next_save = groups + args.save_every
            print(json.dumps({"stage": "rl", "groups": groups, **stats}), flush=True)
        if groups >= args.groups or time.monotonic() >= deadline:
            break
    save_checkpoint(args.output, model, optimizer, groups, "rl", policy, policy_objective=objective)
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
    scope = getattr(args, "expression_scope", None)
    if model.expression_info is not None:
        if scope != model.expression_info["scope"][args.marker.upper()]["scope"]:
            raise ValueError("Specify --expression-scope matching the scorer's labeled input unit")
        if args.mpp is not None:
            raise ValueError("Expression inference resizes the whole input unit and does not accept --mpp")
        from .expression import whole_image

        image, (width, height) = whole_image(source, cfg.crop)
        noise = torch.randn(image.shape, generator=torch.Generator().manual_seed(args.seed))
        with precision(device):
            estimate, probability = sample_rgb(model, image[None].to(device), None, marker,
                                               noise[None].to(device), policy)
        rgb = estimate[0, :, :height, :width].permute(1, 2, 0).cpu().numpy()
        generated = Image.fromarray(np.rint((rgb + 1) * 127.5).astype(np.uint8)).resize(source.size, Image.Resampling.BILINEAR)
        atomic_write(args.output, lambda stream: generated.save(stream, format="PNG"))
        if args.maps:
            _, dab, _ = decompose(np.asarray(generated), cfg, "IHC")
            atomic_write(args.maps, lambda stream: np.savez_compressed(
                stream, dab=dab, expression_probability=np.float32(probability.item()), expression_scope=scope))
        print(json.dumps({"marker": args.marker.upper(), "expression_scope": scope,
                          "expression_probability": probability.item(), "file": str(args.output)}))
        return
    if scope is not None:
        raise ValueError("This checkpoint has no validated expression scorer")
    native_size = source.size
    size = canonical_size(native_size, args.mpp if args.mpp is not None else cfg.pixel_size_um, cfg)
    scaled = source.resize(size, Image.Resampling.BILINEAR) if size != native_size else source
    rgb = np.asarray(scaled)
    _, _, valid = decompose(rgb, cfg, "HE")
    height, width = valid.shape
    tile = cfg.crop
    padded_shape = max(height, tile), max(width, tile)
    image = np.pad(rgb, ((0,padded_shape[0]-height),(0,padded_shape[1]-width),(0,0)), constant_values=255)
    image = torch.from_numpy(image.copy()).permute(2,0,1).float() / 127.5 - 1
    tissue = torch.from_numpy(np.pad(valid, ((0,padded_shape[0]-height),(0,padded_shape[1]-width))))
    noise = torch.randn((3,*padded_shape), generator=torch.Generator().manual_seed(args.seed))
    total = torch.zeros((3,*padded_shape))
    prob_total = torch.zeros(padded_shape)
    weight_total = torch.zeros(padded_shape)
    taper = torch.hann_window(tile, periodic=False).clamp_min(0.05)
    blend = taper[:,None] * taper[None,:]
    for y in offsets(height, tile, tile-cfg.overlap):
        for x in offsets(width, tile, tile-cfg.overlap):
            region = (slice(y,y+tile),slice(x,x+tile))
            image_tile = image[:,region[0],region[1]][None].to(device)
            tissue_tile = tissue[region][None,None].to(device)
            noise_tile = noise[:,region[0],region[1]][None].to(device)
            with precision(device):
                estimate, probability = sample_rgb(model, image_tile, tissue_tile, marker, noise_tile, policy)
            total[:,region[0],region[1]] += estimate.float().cpu()[0] * blend
            prob_total[region] += broadcast_grid(probability, image_tile).float().cpu()[0,0] * blend
            weight_total[region] += blend
    result = (total / weight_total)[:,:height,:width].permute(1,2,0).clamp(-1,1).numpy()
    rendered = Image.fromarray(np.rint((result + 1)*127.5).astype(np.uint8)).resize(native_size, Image.Resampling.BILINEAR)
    atomic_write(args.output, lambda stream: rendered.save(stream, format="PNG"))
    if args.maps:
        _, dab, _ = decompose(np.asarray(rendered), cfg, "IHC")
        probability = Image.fromarray((prob_total / weight_total)[:height,:width].numpy()).resize(native_size, Image.Resampling.BILINEAR)
        atomic_write(args.maps, lambda stream: np.savez_compressed(stream, dab=dab, probability=np.asarray(probability)))
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
