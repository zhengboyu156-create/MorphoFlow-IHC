"""Label-free RGB flow and IHC distribution critic."""

import torch
from torch import nn
import torch.nn.functional as F
from torchcfm import ConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModel

from .config import Config
from .morphology import (LocalMorphologyFPN, canonical_hematoxylin, cosine_distillation,
                         morphology_channels, position_nce)


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
        self.patch_encoder = LocalMorphologyFPN(cfg.morphology_width, cfg.morphology_channels)
        self.teacher_projection = (nn.Conv2d(cfg.morphology_teacher_dim, cfg.morphology_channels, 1)
                                   if cfg.teacher_weight > 0 else None)
        self.critic = IHCCritic(len(cfg.markers))
        self.matcher = ConditionalFlowMatcher(sigma=0.0)

    def velocity(self, t, z, image, marker):
        features = torch.cat((z, image), 1)
        return self.net(t, features, y=marker).float()

    def morphology(self, image, domain):
        return self.patch_encoder(morphology_channels(image, self.cfg, domain))

    def morphology_loss(self, he, generated, image, teacher=None):
        source, _ = self.morphology(he, "HE")
        output, _ = self.morphology(generated, "IHC")
        loss = position_nce(source, output)
        if teacher is None or self.cfg.teacher_weight == 0:
            return loss
        if teacher.output_channels != self.cfg.morphology_teacher_dim:
            raise ValueError("Configured morphology_teacher_dim does not match the loaded teacher")
        real_ihc, _ = self.morphology(image, "IHC")
        canonical = torch.cat((canonical_hematoxylin(he, self.cfg, "HE"),
                               canonical_hematoxylin(image, self.cfg, "IHC")))
        target = self.teacher_projection(teacher(canonical))
        target_he, target_ihc = target.chunk(2)
        distillation = 0.5 * (cosine_distillation(source, target_he.detach())
                              + cosine_distillation(real_ihc, target_ihc.detach()))
        return loss + self.cfg.teacher_weight * distillation

    def losses(self, image, he, marker, teacher=None):
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
        return flow_loss, self.morphology_loss(he, generated, image, teacher), generated


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
            nn.Conv2d(12, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 3, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def observation(self, h, z, velocity, compartments):
        coarse = F.avg_pool2d(torch.cat((h, z, velocity), 1), self.region)
        compartments = F.adaptive_avg_pool2d(compartments, coarse.shape[-2:])
        return torch.cat((coarse, compartments), 1).float()

    def forward(self, state):
        return self.net(state).float().clamp(-5, 5)
