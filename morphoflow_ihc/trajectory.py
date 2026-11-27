"""Paired RGB flow reward and one-step group PPO."""

from copy import deepcopy
import math

import torch
from torch.distributions import Independent, Normal
import torch.nn.functional as F
from torchdiffeq import odeint



def regional_average(values, window):
    """Average over overlapping windows to reduce fixed-grid phase sensitivity.

    Half-window stride makes nearby serial-section shifts affect adjacent
    regions gradually instead of abruptly crossing a non-overlapping cell edge.
    This is regional tolerance, not a substitute for image registration.
    """
    stride = max(1, window // 2)
    return F.avg_pool2d(values.float(), kernel_size=window, stride=stride)


def window_mean(values, selection, window):
    count = regional_average(selection.float(), window)
    values = regional_average(values.float() * selection, window) / count.clamp_min(1e-8)
    present = count > 0
    result = (values * present).flatten(1).sum(1) / present.flatten(1).sum(1).clamp_min(1)
    return result, present.flatten(1).any(1)


def terminal_reward(estimate, reference, tissue, tau, window, intensity_weight):
    """Balance reference hits and abstention on low-DAB tissue by region."""
    estimate, reference = estimate.float(), reference.float()
    positive = (reference > tau) & tissue
    low = (~positive) & tissue
    predicted = estimate > tau
    hits, has_positive = window_mean(predicted, positive, window)
    specificity, has_low = window_mean(~predicted, low, window)
    error, _ = window_mean((estimate - reference).abs(), positive, window)
    intensity = 1 / (1 + error / tau.flatten(1)[:, 0].clamp_min(1e-6))
    balanced = (2 * hits * specificity) / (hits + specificity).clamp_min(1e-8)
    support = torch.where(has_positive & has_low, balanced, torch.where(has_positive, hits, specificity))
    amplitude = torch.where(has_positive, 1 - intensity_weight + intensity_weight * intensity, 1.0)
    return (support * amplitude).clamp(0, 1)


def grid(model, z):
    return torch.linspace(0, 1, model.cfg.ode_steps + 1, device=z.device, dtype=torch.float32)


def integrate(model, h, marker, state, times):
    if times.numel() == 1:
        return state
    return odeint(
        lambda t, z: model.velocity(t, z, h, marker),
        state,
        times,
        method="euler",
    )[-1]


def corrected_velocity(model, base, action):
    # Keep policy control alive when a cold-start flow has near-zero velocity.
    # Positive bilinear weights preserve an L-infinity bound at each pixel.
    rms = torch.linalg.vector_norm(base, dim=(1, 2, 3), keepdim=True) / math.sqrt(base[0].numel())
    scale = rms.clamp_min(model.cfg.velocity_rms_floor)
    residual = model.cfg.velocity_limit * scale * F.interpolate(action.tanh(), size=base.shape[-2:], mode="bilinear", align_corners=False)
    return base + residual


def sample_rgb(model, h, marker, noise, policy=None):
    times = grid(model, noise)
    if policy is None:
        return integrate(model, h, marker, noise, times).clamp(-1, 1)
    k = model.cfg.action_step
    state = integrate(model, h, marker, noise, times[:k + 1])
    base = model.velocity(times[k], state, h, marker)
    observation = policy.observation(h, state, base, model.morphology(h, "HE")[1])
    action = policy(observation)
    initial = state + (times[k + 1] - times[k]) * corrected_velocity(model, base, action)
    result = integrate(model, h, marker, initial, times[k + 1:])
    return result.clamp(-1, 1)


def action_distribution(mean, std):
    return Independent(Normal(mean.float(), std), reinterpreted_batch_ndims=3)


def paired_expression_reward(model, generated, reference, marker):
    """Differentiable regional DAB support/amount reward from paired IHC.

    Regional pooling reduces sensitivity to small serial-section shifts; it
    does not establish exact cell-wise correspondence.
    """
    cfg = model.cfg
    estimated = rgb_to_dab(generated, cfg)
    target = rgb_to_dab(reference, cfg).detach()
    window = cfg.region
    thresholds = generated.new_tensor(list(cfg.markers.values()))[marker].view(-1, 1, 1, 1)
    predicted_support = torch.sigmoid((estimated - thresholds) / 0.05)
    target_support = (target > thresholds).float()
    predicted_fraction = regional_average(predicted_support, window)
    target_fraction = regional_average(target_support, window)
    # Include empty/background windows so extra DAB outside tissue is penalized.
    active = target_fraction > 0.01

    def balanced_error(error):
        positive = (error * active).flatten(1).sum(1) / active.flatten(1).sum(1).clamp_min(1)
        negative_mask = ~active
        negative = (error * negative_mask).flatten(1).sum(1) / negative_mask.flatten(1).sum(1).clamp_min(1)
        has_positive = active.flatten(1).any(1)
        has_negative = negative_mask.flatten(1).any(1)
        return torch.where(has_positive & has_negative, 0.5 * (positive + negative),
                           torch.where(has_positive, positive, negative))

    support_error = balanced_error((predicted_fraction - target_fraction).abs())

    predicted_amount = regional_average(estimated, window)
    target_amount = regional_average(target, window)
    amount_error = balanced_error((predicted_amount - target_amount).abs())
    amount_score = torch.exp(-amount_error / thresholds.flatten(1)[:, 0].clamp_min(1e-6))
    support_score = (1 - support_error).clamp(0, 1)
    weight = cfg.intensity_weight
    return ((1 - weight) * support_score + weight * amount_score).clamp(0, 1)


def collect(model, policy, he, reference, marker, *, direct=False):
    cfg = model.cfg
    noise = torch.randn_like(he)
    times = grid(model, noise)
    k = cfg.action_step
    state = integrate(model, he, marker, noise, times[:k + 1])
    base = model.velocity(times[k], state, he, marker)
    observation = policy.observation(he, state, base, model.morphology(he, "HE")[1])
    mean = policy(observation)
    dist = action_distribution(mean, cfg.action_std)
    actions = dist.rsample((2,)) if direct else dist.sample((2,))
    log_prob = dist.log_prob(actions)
    dt = times[k + 1] - times[k]
    rewards = []
    for action in actions:
        initial = state + dt * corrected_velocity(model, base, action)
        rgb = integrate(model, he, marker, initial, times[k + 1:]).clamp(-1, 1)
        rewards.append(paired_expression_reward(model, rgb, reference, marker))
    return observation, mean, actions, log_prob, torch.stack(rewards)


def action_kl(policy, observation, old_mean, cfg):
    with torch.no_grad():
        delta = policy(observation).float() - old_mean
        return (delta.square() / (2 * cfg.action_std ** 2)).flatten(1).sum(1).mean()


def reward_group_stats(rewards, cfg):
    """Summarize the two-candidate reward signal and flag near-ties."""
    if rewards.ndim != 2 or rewards.shape[0] != 2:
        raise ValueError("Policy updates require two candidate rewards per sample")
    if not torch.isfinite(rewards).all():
        raise FloatingPointError("Nonfinite trajectory reward")
    spread = (rewards[0] - rewards[1]).abs()
    informative = spread >= cfg.min_reward_spread
    wins = (rewards[0] > rewards[1]).float() + 0.5 * (rewards[0] == rewards[1]).float()
    return informative, {
        "reward": rewards.mean().item(),
        "spread": spread.mean().item(),
        "spread_median": spread.median().item(),
        "signal_fraction": informative.float().mean().item(),
        "candidate0_win_rate": wins.mean().item(),
    }


def direct_policy_update(policy, optimizer, experience, cfg):
    observation, old_mean, _, _, rewards = experience
    informative, stats = reward_group_stats(rewards, cfg)
    stats.update({"accepted": 0, "policy_loss": 0.0, "kl": 0.0})
    if not informative.any():
        return stats
    snapshot = deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    mask = informative.to(rewards.dtype).unsqueeze(0)
    loss = -(rewards * mask).sum() / mask.expand_as(rewards).sum().clamp_min(1)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite direct policy loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    kl = action_kl(policy, observation, old_mean.detach(), cfg)
    accepted = bool(torch.isfinite(kl) and kl <= cfg.kl_limit)
    if not accepted:
        policy.load_state_dict(snapshot[0])
        optimizer.load_state_dict(snapshot[1])
    stats.update({"accepted": int(accepted), "policy_loss": loss.item(), "kl": kl.item()})
    return stats


def policy_update(policy, optimizer, experience, cfg, passes=2):
    observation, old_mean, actions, old_log_prob, rewards = experience
    informative, stats = reward_group_stats(rewards, cfg)
    stats.update({"accepted": 0, "policy_loss": 0.0, "kl": 0.0})
    # Group-standardize each sample's two candidate returns. The minimum
    # spread gate prevents tiny numerical differences from becoming unit
    # advantages and driving a noisy policy update.
    advantage = rewards - rewards.mean(dim=0, keepdim=True)
    group_std = rewards.std(dim=0, unbiased=False).clamp_min(cfg.min_reward_spread)
    advantage = (advantage / group_std.unsqueeze(0)).clamp(-1, 1)
    advantage = torch.where(informative.unsqueeze(0), advantage, 0.0)
    if not informative.any():
        return stats
    for _ in range(passes):
        # Revert optimizer moments as well as the weights if a step exceeds the budget.
        before = deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        dist = action_distribution(policy(observation), cfg.action_std)
        ratio = (dist.log_prob(actions) - old_log_prob).clamp(-30, 30).exp()
        surrogate = torch.minimum(
            ratio * advantage,
            ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * advantage,
        )
        # Tied samples are excluded from the signal fraction; they must also
        # be excluded from the denominator, or they silently shrink updates.
        loss = -surrogate[:, informative].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite trajectory policy loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        kl = action_kl(policy, observation[informative], old_mean[informative], cfg)
        stats["policy_loss"], stats["kl"] = loss.item(), kl.item()
        if not torch.isfinite(kl) or kl > cfg.kl_limit:
            policy.load_state_dict(before[0])
            optimizer.load_state_dict(before[1])
            break
        stats["accepted"] += 1
    return stats


def rgb_to_dab(image, cfg):
    rgb = ((image.float()+1)/2).clamp(0, 1)
    od = -torch.log((rgb * 255 + 1) / 256)
    h = F.normalize(od.new_tensor(cfg.stain_h), dim=0)
    d = F.normalize(od.new_tensor(cfg.stain_d), dim=0)
    vectors = torch.stack((h, d), dim=1)
    concentrations = torch.einsum("ij,bjhw->bihw", torch.linalg.pinv(vectors), od)
    boundary = (concentrations < 0).any(1, keepdim=True)
    candidates = torch.einsum("ji,bjhw->bihw", vectors, od).clamp_min(0)
    edge = candidates.argmax(1, keepdim=True)
    edge_dab = torch.where(edge == 1, candidates[:, 1:2], 0.0)
    return torch.where(boundary, edge_dab, concentrations[:, 1:2])
