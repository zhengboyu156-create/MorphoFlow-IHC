"""Training, checkpointing and native-size virtual IHC inference."""

from contextlib import nullcontext
from dataclasses import asdict
import json
import math
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
from .trajectory import collect, direct_policy_update, policy_update, regional_average, sample_rgb, terminal_reward


def choose_device(name):
    device = torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this PyTorch build")
    return device


def amp_dtype(device, mode="auto"):
    if mode not in {"auto", "fp32", "fp16", "bf16"}:
        raise ValueError(f"Unsupported precision mode: {mode}")
    if device.type != "cuda":
        if mode in {"fp16", "bf16"}:
            raise ValueError(f"{mode} autocast requires CUDA")
        return None
    if mode == "fp32":
        return None
    if mode == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 autocast is unsupported by this CUDA device")
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else None


def precision(device, mode="auto"):
    dtype = amp_dtype(device, mode)
    return torch.autocast("cuda", dtype=dtype) if dtype is not None else nullcontext()


def make_grad_scaler(device, mode="auto"):
    dtype = amp_dtype(device, mode)
    if device.type != "cuda" or dtype != torch.float16:
        return None
    return torch.amp.GradScaler("cuda")


CHECKPOINT_FORMAT = 11


def save_checkpoint(path, model, optimizer, step, stage, policy=None, critic_optimizer=None, policy_objective=None,
                    flow_optimizer=None, scaler=None, run=None):
    state = {
        "format": CHECKPOINT_FORMAT, "stage": stage, "step": step, "config": asdict(model.cfg),
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "run": run,
    }
    if critic_optimizer is not None:
        state["critic_optimizer"] = critic_optimizer.state_dict()
    if flow_optimizer is not None:
        state["flow_optimizer"] = flow_optimizer.state_dict()
    if scaler is not None and scaler.is_enabled():
        state["grad_scaler"] = scaler.state_dict()
    if policy is not None:
        state["policy"] = policy.state_dict()
        state["policy_objective"] = policy_objective
    atomic_write(path, lambda stream: torch.save(state, stream))


def open_checkpoint(path, device, *, allow_legacy=False):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format") not in ({9, 10, CHECKPOINT_FORMAT} if allow_legacy else {CHECKPOINT_FORMAT}) or state.get("stage") not in {"flow", "rl"}:
        raise ValueError("Incompatible training checkpoint. Start a fresh format 11 run")
    # Checkpoints written before group-normalized PPO stored a fixed
    # `advantage_scale`; it is obsolete under the current reward-signal gate.
    config = dict(state["config"])
    config.pop("advantage_scale", None)
    # Preserve the two-candidate behavior of pre-v4 policy checkpoints when resuming.
    if state["stage"] == "rl" and "trajectory_candidates" not in config:
        config["trajectory_candidates"] = 2
    config.setdefault("flow_objective", "regional")
    model = StainFlow(Config(**config))
    model.load_state_dict(state["model"])
    model.to(device)
    policy = None
    if state["stage"] == "rl":
        policy = VelocityPolicy(model.cfg)
        if state.get("policy", {}).get("net.4.weight", torch.empty(0)).shape[0] != 3:
            raise ValueError("Incompatible policy checkpoint: retrain the RGB-channel policy")
        policy.load_state_dict(state["policy"])
        policy.to(device)
    return model, policy, state


def batches(loader):
    while True:
        yield from loader


class EpochOrder:
    """Stable pair order and crop seeds, including the first epoch after resume."""

    def __init__(self, size, seed, consumed):
        self.size = size
        self.seed = seed
        self.epoch, self.offset = divmod(consumed, size)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.size, generator=generator)
        seeds = torch.randint(0, (1 << 63) - 1, (self.size,), generator=generator)
        offset = self.offset
        self.epoch += 1
        self.offset = 0
        return iter(zip(order[offset:].tolist(), seeds[offset:].tolist()))


def make_loader(dataset, args, device, consumed):
    return DataLoader(dataset, batch_size=args.batch, sampler=EpochOrder(len(dataset), args.seed, consumed),
                      num_workers=args.workers, pin_memory=device.type == "cuda",
                      persistent_workers=args.workers > 0, generator=torch.Generator().manual_seed(args.seed))


def training_run(args, dataset, device):
    return {"cache_index_sha256": dataset.fingerprint, "seed": args.seed,
            "batch": args.batch, "amp": str(amp_dtype(device, args.amp) or torch.float32)}


def check_resume(state, run):
    if state.get("run") != run:
        raise ValueError("Resume requires the same cache index, seed, batch size, precision and LR schedule")


def flow_optimizers(model):
    parameters = [param for name, param in model.named_parameters() if not name.startswith("critic.")]
    return (torch.optim.AdamW(parameters, lr=model.cfg.flow_lr),
            torch.optim.AdamW(model.critic.parameters(), lr=model.cfg.critic_lr))


def maps_40_10_lr_factor(epoch_index, constant_epochs=40, decay_epochs=10):
    """MAPS-style LR: hold through 40 epochs, then linearly decay over 10."""
    if epoch_index < constant_epochs:
        return 1.0
    return max(0.0, 1.0 - (epoch_index - constant_epochs + 1) / (decay_epochs + 1))


def flow_update(model, optimizer, critic_optimizer, image, he, marker, policy=None, scaler=None, amp="auto"):
    cfg = model.cfg
    critic_enabled = cfg.adversarial_weight > 0
    scale_before = scaler.get_scale() if scaler is not None and scaler.is_enabled() else None
    optimizer.zero_grad(set_to_none=True)
    if critic_enabled:
        critic_optimizer.zero_grad(set_to_none=True)
    policy_grad_state = [p.requires_grad for p in policy.parameters()] if policy is not None else None
    if policy is not None:
        # The policy is optimized by PPO/direct updates; here its action is a
        # fixed controller, while gradients still pass through its inputs to G.
        for parameter in policy.parameters():
            parameter.grad = None
        policy.requires_grad_(False)
    try:
        with precision(he.device, amp):
            flow_loss, nce_loss, fake = model.losses(image, he, marker, policy=policy)
            if critic_enabled:
                critic_loss = 0.5 * (F.softplus(-model.critic(image, marker)).mean()
                                     + F.softplus(model.critic(fake.detach(), marker)).mean())
            else:
                critic_loss = flow_loss.new_zeros(())
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Nonfinite critic loss")
        if critic_enabled:
            (scaler.scale(critic_loss) if scaler is not None else critic_loss).backward()
            if scaler is not None:
                scaler.unscale_(critic_optimizer)
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0,
                                          error_if_nonfinite=scaler is None)
            if scaler is not None:
                scaler.step(critic_optimizer)
            else:
                critic_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)

        model.critic.eval().requires_grad_(False)
        try:
            with precision(he.device, amp):
                adversarial_loss = (F.softplus(-model.critic(fake, marker)).mean()
                                    if critic_enabled else flow_loss.new_zeros(()))
                loss = flow_loss + cfg.nce_weight * nce_loss + cfg.adversarial_weight * adversarial_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite flow training loss")
            (scaler.scale(loss) if scaler is not None else loss).backward()
            if scaler is not None:
                scaler.unscale_(optimizer)
            parameters = [param for group in optimizer.param_groups for param in group["params"]]
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=scaler is None)
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        finally:
            model.critic.train().requires_grad_(True)
    finally:
        if policy is not None:
            for parameter, requires_grad in zip(policy.parameters(), policy_grad_state):
                parameter.requires_grad_(requires_grad)
    stats = {"flow_loss": flow_loss.item(), "nce_loss": nce_loss.item(),
             "critic_loss": critic_loss.item(), "adversarial_loss": adversarial_loss.item()}
    if scale_before is not None:
        stats["amp_scale"] = scaler.get_scale()
        stats["amp_overflow"] = int(stats["amp_scale"] < scale_before)
    return stats


def train_flow(args, cfg):
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    scaler = make_grad_scaler(device, args.amp)
    step = 0
    if args.resume:
        model, _, state = open_checkpoint(args.resume, device)
        if state["stage"] != "flow" or model.cfg != cfg:
            raise ValueError("Resume requires a flow checkpoint with the same model configuration")
        step = state["step"]
    else:
        model = StainFlow(cfg).to(device)
    dataset = IHCCrops(args.cache, cfg)
    run = training_run(args, dataset, device)
    updates_per_epoch = math.ceil(len(dataset) / args.batch)
    epoch, offset = divmod(step, updates_per_epoch)
    loader = make_loader(dataset, args, device, epoch * len(dataset) + offset * args.batch)
    optimizer, critic_optimizer = flow_optimizers(model)
    if args.resume:
        check_resume(state, run)
        optimizer.load_state_dict(state["optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
        if scaler is not None and "grad_scaler" in state:
            scaler.load_state_dict(state["grad_scaler"])
    target = args.steps or args.epochs * updates_per_epoch
    if step > target:
        raise ValueError("Flow checkpoint already exceeds the requested update target")
    print(json.dumps({"stage": "flow", "event": "start", "step": step, "target": target,
                      "amp": args.amp, "run": run}), flush=True)
    deadline = None if args.no_time_limit else time.monotonic() + 3600 * args.hours
    for pack in batches(loader):
        if step >= target or (deadline is not None and time.monotonic() >= deadline):
            break
        marker, image, he = (value.to(device, non_blocking=True) for value in pack)
        torch.manual_seed(args.seed + step)
        stats = flow_update(model, optimizer, critic_optimizer, image, he, marker, scaler=scaler, amp=args.amp)
        step += 1
        if step == 1 or step % args.save_every == 0 or step == target:
            save_checkpoint(args.output, model, optimizer, step, "flow", critic_optimizer=critic_optimizer,
                            scaler=scaler, run=run)
            print(json.dumps({"stage": "flow", "step": step, **stats}), flush=True)
        if step >= target or (deadline is not None and time.monotonic() >= deadline):
            break
    save_checkpoint(args.output, model, optimizer, step, "flow", critic_optimizer=critic_optimizer, scaler=scaler, run=run)
    print(json.dumps({"stage": "flow", "step": step, "target": target, "file": str(args.output)}))


def train_policy(args):
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    model, policy, state = open_checkpoint(args.checkpoint, device)
    scaler = make_grad_scaler(device, args.amp)
    if scaler is not None and "grad_scaler" in state:
        scaler.load_state_dict(state["grad_scaler"])
    cfg = model.cfg
    if args.policy_lr is not None:
        cfg.rl_lr = args.policy_lr
    flow_optimizer, critic_optimizer = flow_optimizers(model)
    if state["stage"] == "rl" and "flow_optimizer" not in state:
        raise ValueError("Start joint training from a base flow checkpoint or resume a joint checkpoint")
    flow_optimizer.load_state_dict(state["flow_optimizer"] if state["stage"] == "rl" else state["optimizer"])
    critic_optimizer.load_state_dict(state["critic_optimizer"])
    if policy is None:
        policy = VelocityPolicy(cfg).to(device)
    policy_lr = cfg.rl_lr if args.policy_lr is None else args.policy_lr
    optimizer = torch.optim.AdamW(policy.parameters(), lr=policy_lr)
    groups = 0
    objective = args.objective
    if state["stage"] == "rl":
        if (state.get("policy_objective") or "ppo") != objective:
            raise ValueError("Start both objectives from the same base flow checkpoint")
        groups = state["step"]
        optimizer.load_state_dict(state["optimizer"])
        if args.policy_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = policy_lr
    dataset = IHCCrops(args.cache, cfg)
    run = {**training_run(args, dataset, device), "decay_epochs": args.epochs,
           "zero_action": args.zero_action}
    if (state.get("run") or {}).get("cache_index_sha256") != dataset.fingerprint:
        raise ValueError("Joint training requires the same training cache as the Flow checkpoint")
    if state["stage"] == "rl":
        check_resume(state, run)
    loader = make_loader(dataset, args, device, groups)
    target_groups = args.groups if args.groups is not None else args.epochs * len(dataset)
    if groups > target_groups or target_groups > args.epochs * len(dataset):
        raise ValueError("Group target must cover the resume position and fit within the LR decay epochs")
    print(json.dumps({"stage": "rl", "event": "start", "groups": groups, "target": target_groups,
                      "amp": args.amp, "run": run}), flush=True)
    next_save = groups + args.save_every
    deadline = None if args.no_time_limit else time.monotonic() + 3600 * args.hours
    for pack in batches(loader):
        if groups >= target_groups or (deadline is not None and time.monotonic() >= deadline):
            break
        limit = target_groups - groups
        marker, image, he = (value[:limit].to(device, non_blocking=True) for value in pack)
        torch.manual_seed(args.seed + groups)
        epoch_index = 40 + groups // len(dataset)
        lr_factor = maps_40_10_lr_factor(epoch_index, decay_epochs=args.epochs)
        for current_optimizer, base_lr in (
            (flow_optimizer, cfg.flow_lr),
            (critic_optimizer, cfg.critic_lr),
            (optimizer, policy_lr),
        ):
            for param_group in current_optimizer.param_groups:
                param_group["lr"] = base_lr * lr_factor
        direct = objective == "direct"
        # Keep this rollout's environment fixed through its policy update.
        model.eval().requires_grad_(False)
        try:
            if args.zero_action:
                stats = {"reward": 0.0, "spread": 0.0, "spread_median": 0.0,
                         "signal_fraction": 0.0, "candidate0_win_rate": 0.0,
                         "accepted": 0, "policy_loss": 0.0, "kl": 0.0}
            else:
                with precision(device, args.amp), torch.set_grad_enabled(direct):
                    experience = collect(model, policy, he, image, marker, direct=direct)
                if direct:
                    stats = direct_policy_update(policy, optimizer, experience, cfg, scaler=scaler)
                else:
                    stats = policy_update(policy, optimizer, experience, cfg)
        finally:
            model.train().requires_grad_(True)
        # Keep generator-side randomness matched between PPO and zero-action arms.
        torch.manual_seed(args.seed + groups)
        stats.update(flow_update(model, flow_optimizer, critic_optimizer, image, he, marker,
                                 policy=None if args.zero_action else policy,
                                 scaler=scaler, amp=args.amp))
        stats["objective"] = objective
        stats["lr_factor"] = lr_factor
        stats["policy_lr"] = optimizer.param_groups[0]["lr"]
        stats["flow_lr"] = flow_optimizer.param_groups[0]["lr"]
        stats["critic_lr"] = critic_optimizer.param_groups[0]["lr"]
        groups += he.shape[0]
        if groups >= next_save or groups == args.groups or groups == he.shape[0]:
            save_checkpoint(args.output, model, optimizer, groups, "rl", policy, critic_optimizer=critic_optimizer,
                            policy_objective=objective, flow_optimizer=flow_optimizer, scaler=scaler, run=run)
            next_save = groups + args.save_every
            print(json.dumps({"stage": "rl", "groups": groups, **stats}), flush=True)
        if groups >= target_groups or (deadline is not None and time.monotonic() >= deadline):
            break
    save_checkpoint(args.output, model, optimizer, groups, "rl", policy, critic_optimizer=critic_optimizer,
                            policy_objective=objective, flow_optimizer=flow_optimizer, scaler=scaler, run=run)
    print(json.dumps({"stage": "rl", "groups": groups, "target": target_groups, "file": str(args.output)}))


@torch.no_grad()
def infer(args):
    device = choose_device(args.device)
    model, policy, _ = open_checkpoint(args.checkpoint, device, allow_legacy=True)
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
    downsample_factor = 2 ** (len(cfg.channel_mult) - 1)
    if rgb.shape[0] % downsample_factor or rgb.shape[1] % downsample_factor:
        raise ValueError(
            f"Full-image inference dimensions must be divisible by {downsample_factor}; "
            f"got {rgb.shape[1]}x{rgb.shape[0]}. Use native 1024x1024 benchmark images."
        )
    image = torch.from_numpy(rgb.copy()).permute(2,0,1).float() / 127.5 - 1
    image = image[None].to(device)
    noise = torch.randn(image.shape[1:], generator=torch.Generator().manual_seed(args.seed))[None].to(device)
    with precision(device, "fp32"):
        estimate = sample_rgb(model, image, marker, noise, policy)
    if not torch.isfinite(estimate).all():
        raise FloatingPointError("Nonfinite model output")
    result = estimate.float().cpu()[0].permute(1,2,0).clamp(-1,1).numpy()
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
    # Pad for the coarser reward scale; it is also divisible by `region`.
    pad = (0, -width % cfg.reward_region, 0, -height % cfg.reward_region)
    estimate, actual = (F.pad(torch.from_numpy(a.astype(np.float32))[None, None], pad) for a in (estimated, true_dab))
    seen = F.pad(torch.from_numpy(tissue)[None, None], pad)
    if not seen.any():
        raise ValueError("Reference contains no evaluable tissue")
    def phase_support_error(window):
        count_phases = regional_average(seen.float(), window)
        support_phases = [regional_average(((a > tau) & seen).float(), window)
                          for a in (estimate, actual)]
        errors = []
        for count, estimated_support, actual_support in zip(
            count_phases, support_phases[0], support_phases[1]
        ):
            estimated_fraction = estimated_support / count.clamp_min(1e-8)
            actual_fraction = actual_support / count.clamp_min(1e-8)
            valid = count > 0
            errors.append((estimated_fraction - actual_fraction).abs()[valid].mean())
        return torch.stack(errors).mean().item()

    metrics = {
        "marker": args.marker.upper(), "threshold": tau, "region_px": region,
        "region_support_fraction_mae": phase_support_error(region),
        "reference_tissue_fraction": seen.float().sum().item() / (height * width),
    }
    metrics["reward_region_px"] = cfg.reward_region
    metrics["reward_region_support_fraction_mae"] = phase_support_error(cfg.reward_region)
    if args.same_section:
        metrics["pixel_dab_mae"] = (estimate - actual).abs()[seen].mean().item()
        metrics["pixel_support_reward"] = terminal_reward(
            estimate, actual, seen, torch.tensor([[[[tau]]]]), cfg.reward_region, cfg.intensity_weight
        ).item()
    report = json.dumps(metrics, indent=2, allow_nan=False)
    if args.output:
        atomic_write(args.output, lambda stream: stream.write(report.encode()))
    print(report)
