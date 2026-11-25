"""RGB flow, unpaired terminal reward, and one-step group PPO."""

from copy import deepcopy
import math

import torch
from torch.distributions import Independent, Normal
import torch.nn.functional as F
from torchdiffeq import odeint



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
    # Positive bilinear weights preserve an L-infinity bound at each pixel.
    rms = torch.linalg.vector_norm(base, dim=(1, 2, 3), keepdim=True) / math.sqrt(base[0].numel())
    residual = model.cfg.velocity_limit * rms * F.interpolate(action.tanh(), size=base.shape[-2:], mode="bilinear", align_corners=False)
    return base + residual


def sample_rgb(model, h, marker, noise, policy=None):
    times = grid(model, noise)
    if policy is None:
        return integrate(model, h, marker, noise, times).clamp(-1, 1)
    k = model.cfg.action_step
    state = integrate(model, h, marker, noise, times[:k + 1])
    base = model.velocity(times[k], state, h, marker)
    observation = policy.observation(h, state, base)
    action = policy(observation)
    initial = state + (times[k + 1] - times[k]) * corrected_velocity(model, base, action)
    result = integrate(model, h, marker, initial, times[k + 1:])
    return result.clamp(-1, 1)


def action_distribution(mean, std):
    return Independent(Normal(mean.float(), std), reinterpreted_batch_ndims=3)


def unpaired_reward(model, he, generated, marker, indices):
    """Rate target-domain realism and spatial content on each candidate."""
    endpoint = model.critic(generated, marker).sigmoid().flatten(1).mean(1)
    source = model.patch_encoder(he).flatten(2).transpose(1, 2)[:, indices]
    output = model.patch_encoder(generated).flatten(2).transpose(1, 2)[:, indices]
    source = F.normalize(source.float(), dim=-1)
    output = F.normalize(output.float(), dim=-1)
    logits = torch.bmm(output, source.transpose(1, 2)) / 0.07
    labels = torch.arange(len(indices), device=he.device).expand(he.shape[0], -1)
    xent = F.cross_entropy(logits.flatten(0, 1), labels.reshape(-1), reduction="none").reshape(he.shape[0], -1).mean(1)
    structure = torch.exp(-xent / max(1.0, math.log(len(indices))))
    return (0.5 * endpoint + 0.5 * structure).clamp(0, 1)


@torch.no_grad()
def reward_indices(model, he):
    height, width = model.patch_encoder(he).shape[-2:]
    return torch.randperm(height * width, device=he.device)[:64]


def collect(model, policy, he, marker, *, direct=False):
    cfg = model.cfg
    noise = torch.randn_like(he)
    times = grid(model, noise)
    k = cfg.action_step
    state = integrate(model, he, marker, noise, times[:k + 1])
    base = model.velocity(times[k], state, he, marker)
    observation = policy.observation(he, state, base)
    mean = policy(observation)
    dist = action_distribution(mean, cfg.action_std)
    actions = dist.rsample((2,)) if direct else dist.sample((2,))
    log_prob = dist.log_prob(actions)
    dt = times[k + 1] - times[k]
    # Share sampled locations between both candidate trajectories.
    indices = reward_indices(model, he)
    rewards = []
    for action in actions:
        initial = state + dt * corrected_velocity(model, base, action)
        rgb = integrate(model, he, marker, initial, times[k + 1:]).clamp(-1, 1)
        rewards.append(unpaired_reward(model, he, rgb, marker, indices))
    return observation, mean, actions, log_prob, torch.stack(rewards)


def action_kl(policy, observation, old_mean, cfg):
    with torch.no_grad():
        delta = policy(observation).float() - old_mean
        return (delta.square() / (2 * cfg.action_std ** 2)).flatten(1).sum(1).mean()


def direct_policy_update(policy, optimizer, experience, cfg):
    observation, old_mean, _, _, rewards = experience
    snapshot = deepcopy(policy.state_dict()), deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    loss = -rewards.mean()
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
    return {"reward": rewards.mean().item(), "spread": (rewards[0] - rewards[1]).abs().mean().item(),
            "accepted": int(accepted), "policy_loss": loss.item(), "kl": kl.item()}


def policy_update(policy, optimizer, experience, cfg, passes=2):
    observation, old_mean, actions, old_log_prob, rewards = experience
    advantage = rewards - rewards.mean(dim=0, keepdim=True)
    stats = {"reward": rewards.mean().item(), "spread": (rewards[0] - rewards[1]).abs().mean().item(),
             "accepted": 0, "policy_loss": 0.0, "kl": 0.0}
    # Ignore numerical ties in the FP32 terminal score.
    advantage = torch.where(advantage.abs() > 1e-6, advantage, 0.0)
    advantage = (advantage / cfg.advantage_scale).clamp(-1, 1)
    if not advantage.abs().any():
        return stats
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
        kl = action_kl(policy, observation, old_mean, cfg)
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
