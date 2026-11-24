"""Morphology-conditioned region prediction and single-channel flow."""

import torch
from torch import nn
import torch.nn.functional as F
from torchcfm import ConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModel

from .config import Config


def broadcast_grid(grid, image):
    return F.interpolate(grid, size=image.shape[-2:], mode="bilinear", align_corners=False)


class RegionPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.region = cfg.region
        self.context = cfg.context
        self.marker = nn.Embedding(len(cfg.markers), 8)
        self.context_head = nn.Sequential(nn.Conv2d(11, 24, 1), nn.SiLU(), nn.Conv2d(24, 1, 1))
        self.local_features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.SiLU(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.detail_head = nn.Sequential(nn.Conv2d(75, 32, 1), nn.SiLU(), nn.Conv2d(32, 1, 1))

    def forward(self, h, tissue, marker):
        batch, _, height, width = h.shape
        region = self.region
        rows, cols = height // region, width // region
        h = h.float()
        valid = tissue.float()
        # Pointwise moments keep center information out of the ring branch.
        moments = torch.cat((h * valid, h.square() * valid, valid), dim=1)
        center = F.avg_pool2d(moments, region) * region ** 2
        pad_before = (self.context - region) // 2
        pad_after = self.context - region - pad_before
        padded = F.pad(moments, (pad_before, pad_after, pad_before, pad_after))
        wide = F.avg_pool2d(padded, self.context, stride=region) * self.context ** 2
        ring = wide - center
        mean = ring[:, :1] / ring[:, 2:3].clamp_min(1)
        mean_sq = ring[:, 1:2] / ring[:, 2:3].clamp_min(1)
        std = (mean_sq - mean.square()).clamp_min(1e-8).sqrt()
        occupancy = ring[:, 2:3].clamp_min(0) / (self.context ** 2 - region ** 2)
        marker_features = self.marker(marker)[:, :, None, None].expand(-1, -1, rows, cols)
        context = torch.cat((mean, std, occupancy, marker_features), dim=1)
        base = self.context_head(context)
        # Each local encoder only sees its own center tile.
        tiles = h.unfold(2, region, region).unfold(3, region, region)
        tiles = tiles.permute(0, 2, 3, 1, 4, 5).reshape(-1, 1, region, region)
        detail = self.local_features(tiles).reshape(batch, rows, cols, 64).permute(0, 3, 1, 2)
        return base, base + self.detail_head(torch.cat((context, detail), dim=1))


class StainFlow(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.region = RegionPredictor(cfg)
        self.net = UNetModel(
            image_size=cfg.crop,
            in_channels=3,
            out_channels=1,
            model_channels=cfg.channels,
            channel_mult=cfg.channel_mult,
            num_res_blocks=cfg.residual_blocks,
            attention_resolutions=(),
            num_classes=len(cfg.markers),
        )
        self.matcher = ConditionalFlowMatcher(sigma=0.0)
        self.register_buffer("thresholds", torch.tensor(list(cfg.markers.values())))

    def probability(self, h, tissue, marker):
        _, logits = self.region(h, tissue, marker)
        return logits.float().sigmoid()

    def velocity(self, t, z, h, probability, marker):
        features = torch.cat((z, h, broadcast_grid(probability, h)), 1)
        return self.net(t, features, y=marker).float()

    def losses(self, h, reference, tissue, marker):
        base, logits = self.region(h, tissue, marker)
        tau = self.thresholds[marker, None, None, None]
        valid = tissue.float()
        occupancy = F.avg_pool2d(valid, self.cfg.region)
        target = F.avg_pool2d(((reference > tau) & tissue).float(), self.cfg.region)
        target = target / occupancy.clamp_min(1e-8)
        # A single proper scoring loss supervises both available views.
        predictions = torch.cat((logits.float(), base.float()), dim=1)
        bce = F.binary_cross_entropy_with_logits(predictions, target.expand_as(predictions), reduction="none")
        weights = predictions.new_tensor([0.75, 0.25]).view(1, 2, 1, 1)
        regions = (occupancy > 0).float()
        region_loss = (bce * weights * regions).sum() / regions.sum().clamp_min(1)
        t, z, target_velocity = self.matcher.sample_location_and_conditional_flow(
            torch.randn_like(reference), reference
        )
        prob = logits.float().sigmoid().detach()
        prediction = self.velocity(t, z, h, prob, marker)
        flow_loss = F.mse_loss(prediction, target_velocity)
        return region_loss, flow_loss


class VelocityPolicy(nn.Module):
    """Diagonal Gaussian action on the region grid with bounded velocity effect."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.region = cfg.region
        self.net = nn.Sequential(
            nn.Conv2d(4, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 1, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def observation(self, h, z, velocity, probability):
        coarse = F.avg_pool2d(torch.cat((h, z, velocity), 1), self.region)
        return torch.cat((coarse, probability), 1).float()

    def forward(self, state):
        return self.net(state).float().clamp(-5, 5)
