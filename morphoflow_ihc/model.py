"""Label-free RGB flow and IHC distribution critic."""

import torch
from torch import nn
import torch.nn.functional as F
from torchcfm import ConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModel

from .config import Config


class StainFlow(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.net = UNetModel(
            image_size=cfg.crop,
            in_channels=6,
            out_channels=3,
            model_channels=cfg.channels,
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.residual_blocks,
            attention_resolutions=(),
            num_classes=len(cfg.markers),
        )
        self.patch_encoder = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(), nn.Conv2d(64, 64, 3, stride=2, padding=1))
        self.critic = IHCCritic(len(cfg.markers))
        self.matcher = ConditionalFlowMatcher(sigma=0.0)
    def velocity(self, t, z, image, marker):
        features = torch.cat((z, image), 1)
        return self.net(t, features, y=marker).float()

    def losses(self, image, he, marker):
        t, z, target_velocity = self.matcher.sample_location_and_conditional_flow(
            torch.randn_like(image), image
        )
        prediction = self.velocity(t, z, torch.zeros_like(image), marker)
        flow_loss = F.mse_loss(prediction, target_velocity)

        # Train the H&E path using the same grid used for generation and policy rollout.
        state = torch.randn_like(he)
        steps = self.cfg.ode_steps
        for i in range(steps):
            step_t = torch.full((he.shape[0],), i / steps, device=he.device)
            state = state + self.velocity(step_t, state, he, marker) / steps
        generated = state.clamp(-1, 1)
        return flow_loss, patch_nce(self.patch_encoder, he, generated), generated


class IHCCritic(nn.Module):
    """Marker-conditioned patch discriminator for unpaired IHC realism."""
    def __init__(self, markers):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.LeakyReLU(0.2),
        )
        self.score = nn.Conv2d(128, 1, 1)
        self.marker = nn.Embedding(markers, 128)

    def forward(self, image, marker):
        features = self.features(image.float())
        projection = (features * self.marker(marker)[:,:,None,None]).sum(1, keepdim=True) / (128 ** 0.5)
        return self.score(features) + projection


class VelocityPolicy(nn.Module):
    """Diagonal RGB Gaussian actions on the region grid with bounded velocity effect."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.region = cfg.region
        self.net = nn.Sequential(
            nn.Conv2d(9, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 3, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def observation(self, h, z, velocity):
        return F.avg_pool2d(torch.cat((h, z, velocity), 1), self.region).float()

    def forward(self, state):
        return self.net(state).float().clamp(-5, 5)


def patch_nce(encoder, source, generated, patches=64, temperature=0.07):
    a = encoder(source).flatten(2).transpose(1, 2)
    b = encoder(generated).flatten(2).transpose(1, 2)
    count = min(patches, a.shape[1])
    indices = torch.randperm(a.shape[1], device=a.device)[:count]
    # Same indices are positives; negatives come from other locations of the same image.
    a = F.normalize(a[:, indices], dim=-1)
    b = F.normalize(b[:, indices], dim=-1)
    logits = torch.bmm(b, a.transpose(1, 2)) / temperature
    target = torch.arange(count, device=a.device).expand(a.shape[0], -1)
    return F.cross_entropy(logits.reshape(-1, count), target.reshape(-1))
