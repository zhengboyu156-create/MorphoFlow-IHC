"""Fixed Euler integration, DAB terminal reward, and one-step group PPO."""

from copy import deepcopy

import torch
from torch.distributions import Independent, Normal, kl_divergence
import torch.nn.functional as F
from torchdiffeq import odeint

from .model import broadcast_grid


def window_mean(values, selection, window):
    count = F.avg_pool2d(selection.float(), window)
    values = F.avg_pool2d(values.float() * selection, window) / count.clamp_min(1e-8)
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


def integrate(model, h, probability, marker, state, times):
    if times.numel() == 1:
        return state
    return odeint(
        lambda t, z: model.velocity(t, z, h, probability, marker),
        state,
        times,
        method="euler",
    )[-1]


def corrected_velocity(model, base, action):
    # Positive bilinear weights preserve an L-infinity bound at each pixel.
    rms = base.square().mean((1, 2, 3), keepdim=True).sqrt()
    residual = model.cfg.velocity_limit * rms * broadcast_grid(action.tanh(), base)
    return base + residual


def sample_dab(model, h, tissue, marker, noise, policy=None):
    probability = model.probability(h, tissue, marker)
    times = grid(model, noise)
    if policy is None:
        return integrate(model, h, probability, marker, noise, times).clamp_min(0), probability
    k = model.cfg.action_step
    state = integrate(model, h, probability, marker, noise, times[:k + 1])
    base = model.velocity(times[k], state, h, probability, marker)
    observation = policy.observation(h, state, base, probability)
    action = policy(observation)
    initial = state + (times[k + 1] - times[k]) * corrected_velocity(model, base, action)
    result = integrate(model, h, probability, marker, initial, times[k + 1:])
    return result.clamp_min(0), probability


def action_distribution(mean, std):
    return Independent(Normal(mean.float(), std), reinterpreted_batch_ndims=3)


@torch.no_grad()
def collect(model, policy, h, reference, tissue, marker):
    cfg = model.cfg
    noise = torch.randn_like(reference)
    probability = model.probability(h, tissue, marker)
    times = grid(model, noise)
    k = cfg.action_step
    state = integrate(model, h, probability, marker, noise, times[:k + 1])
    base = model.velocity(times[k], state, h, probability, marker)
    observation = policy.observation(h, state, base, probability)
    mean = policy(observation)
    dist = action_distribution(mean, cfg.action_std)
    actions = dist.sample((2,))
    log_prob = dist.log_prob(actions)
    dt = times[k + 1] - times[k]
    tau = model.thresholds[marker, None, None, None]
    rewards = []
    # Sequential branches minimize peak memory and share the entire prefix.
    for action in actions:
        initial = state + dt * corrected_velocity(model, base, action)
        dab = integrate(model, h, probability, marker, initial, times[k + 1:]).clamp_min(0)
        rewards.append(terminal_reward(dab, reference, tissue, tau, cfg.region, cfg.intensity_weight))
    return observation, mean, actions, log_prob, torch.stack(rewards)


def policy_update(policy, optimizer, experience, cfg, passes=2):
    observation, old_mean, actions, old_log_prob, rewards = experience
    advantage = rewards - rewards.mean(dim=0, keepdim=True)
    stats = {"reward": rewards.mean().item(), "spread": (rewards[0] - rewards[1]).abs().mean().item(),
             "accepted": 0, "policy_loss": 0.0, "kl": 0.0}
    # Ignore numerical ties in the FP32 terminal score.
    advantage = torch.where(advantage.abs() > 1e-6, advantage, 0.0)
    if not advantage.abs().any():
        return stats
    old = action_distribution(old_mean, cfg.action_std)
    for _ in range(passes):
        # Revert optimizer moments as well as the weights if a step exceeds the budget.
        before = deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        dist = action_distribution(policy(observation), cfg.action_std)
        ratio = (dist.log_prob(actions) - old_log_prob).clamp(-30, 30).exp()
        loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * advantage).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite trajectory policy loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            kl = kl_divergence(old, action_distribution(policy(observation), cfg.action_std)).mean()
        stats["policy_loss"], stats["kl"] = loss.item(), kl.item()
        if not torch.isfinite(kl) or kl > cfg.kl_limit:
            policy.load_state_dict(before[0])
            optimizer.load_state_dict(before[1])
            break
        stats["accepted"] += 1
    return stats
