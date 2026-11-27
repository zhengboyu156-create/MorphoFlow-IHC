"""Paired RGB flow reward and one-step group PPO."""

from copy import deepcopy
import math

import torch
from torch.distributions import Independent, Normal
import torch.nn.functional as F
from torchdiffeq import odeint



def regional_average(values, window):
    """Return non-overlapping regional averages over four half-window offsets.

    Each grid uses stride == window. Averaging scores over shifted grid phases
    reduces sensitivity to one arbitrary boundary without overlapping windows
    inside any phase. This does not replace image registration.
    """
    values = values.float()
    height, width = values.shape[-2:]
    half = max(1, window // 2)
    phases = []
    for top, left in ((0, 0), (0, half), (half, 0), (half, half)):
        bottom = (-(height + top)) % window
        right = (-(width + left)) % window
        padded = F.pad(values, (left, right, top, bottom))
        phases.append(F.avg_pool2d(padded, kernel_size=window, stride=window))
    return phases


def window_mean(values, selection, window):
    counts = regional_average(selection.float(), window)
    totals = regional_average(values.float() * selection, window)
    results = []
    for count, total in zip(counts, totals):
        values = total / count.clamp_min(1e-8)
        present = count > 0
        result = (values * present).flatten(1).sum(1) / present.flatten(1).sum(1).clamp_min(1)
        results.append((result, present.flatten(1).any(1)))
    return results


def terminal_reward(estimate, reference, tissue, tau, window, intensity_weight):
    """Balance reference hits and abstention on low-DAB tissue by region."""
    estimate, reference = estimate.float(), reference.float()
    positive = (reference > tau) & tissue
    low = (~positive) & tissue
    predicted = estimate > tau
    hits = window_mean(predicted, positive, window)
    specificity = window_mean(~predicted, low, window)
    error = window_mean((estimate - reference).abs(), positive, window)
    rewards = []
    for (hit, has_positive), (specific, has_low), (amount_error, _) in zip(hits, specificity, error):
        intensity = 1 / (1 + amount_error / tau.flatten(1)[:, 0].clamp_min(1e-6))
        balanced = (2 * hit * specific) / (hit + specific).clamp_min(1e-8)
        support = torch.where(has_positive & has_low, balanced, torch.where(has_positive, hit, specific))
        amplitude = torch.where(has_positive, 1 - intensity_weight + intensity_weight * intensity, 1.0)
        rewards.append((support * amplitude).clamp(0, 1))
    return torch.stack(rewards).mean(0)


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

    Coarse regional pooling reduces sensitivity to local serial-section shifts;
    it does not establish exact cell-wise correspondence or correct large warps.
    """
    cfg = model.cfg
    estimated = rgb_to_dab(generated, cfg)
    target = rgb_to_dab(reference, cfg).detach()
    window = cfg.reward_region
    thresholds = generated.new_tensor(list(cfg.markers.values()))[marker].view(-1, 1, 1, 1)
    predicted_support = torch.sigmoid((estimated - thresholds) / 0.05)
    target_support = (target > thresholds).float()
    predicted_fractions = regional_average(predicted_support, window)
    target_fractions = regional_average(target_support, window)
    # Include empty/background windows so extra DAB outside tissue is penalized.
    predicted_amounts = regional_average(estimated, window)
    target_amounts = regional_average(target, window)

    def balanced_error(error, active):
        positive = (error * active).flatten(1).sum(1) / active.flatten(1).sum(1).clamp_min(1)
        negative_mask = ~active
        negative = (error * negative_mask).flatten(1).sum(1) / negative_mask.flatten(1).sum(1).clamp_min(1)
        has_positive = active.flatten(1).any(1)
        has_negative = negative_mask.flatten(1).any(1)
        return torch.where(has_positive & has_negative, 0.5 * (positive + negative),
                           torch.where(has_positive, positive, negative))

    scores = []
    for predicted_fraction, target_fraction, predicted_amount, target_amount in zip(
        predicted_fractions, target_fractions, predicted_amounts, target_amounts
    ):
        active = target_fraction > 0.01
        support_error = balanced_error((predicted_fraction - target_fraction).abs(), active)
        amount_error = balanced_error((predicted_amount - target_amount).abs(), active)
        amount_score = torch.exp(-amount_error / thresholds.flatten(1)[:, 0].clamp_min(1e-6))
        support_score = (1 - support_error).clamp(0, 1)
        weight = cfg.intensity_weight
        scores.append(((1 - weight) * support_score + weight * amount_score).clamp(0, 1))
    return torch.stack(scores).mean(0)


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
    actions = dist.rsample((cfg.trajectory_candidates,)) if direct else dist.sample((cfg.trajectory_candidates,))
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
    """Summarize within-state candidate rewards and flag near-ties."""
    if rewards.ndim != 2 or rewards.shape[0] < 2:
        raise ValueError("Policy updates require at least two candidate rewards per sample")
    if not torch.isfinite(rewards).all():
        raise FloatingPointError("Nonfinite trajectory reward")
    spread = rewards.amax(dim=0) - rewards.amin(dim=0)
    informative = spread >= cfg.min_reward_spread
    best = rewards == rewards.amax(dim=0, keepdim=True)
    wins = best.float() / best.sum(dim=0, keepdim=True).clamp_min(1)
    return informative, {
        "reward": rewards.mean().item(),
        "spread": spread.mean().item(),
        "spread_median": spread.median().item(),
        "signal_fraction": informative.float().mean().item(),
        "candidate0_win_rate": wins.mean().item(),
    }


def direct_policy_update(policy, optimizer, experience, cfg, scaler=None):
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
    (scaler.scale(loss) if scaler is not None else loss).backward()
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=scaler is None)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
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
    # Group-standardize each sample's candidate returns. The minimum
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
