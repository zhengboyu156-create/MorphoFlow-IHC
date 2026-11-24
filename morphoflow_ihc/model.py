"""Unpaired RGB flow, morphology conditioning, and IHC distribution critic."""

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
        self.expression_info = None
        self.expression_scorers = nn.ModuleDict()
        self.region = RegionPredictor(cfg)
        self.net = UNetModel(
            image_size=cfg.crop,
            in_channels=7,
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
        self.register_buffer("thresholds", torch.tensor(list(cfg.markers.values())))

    def attach_expression(self, artifact):
        from .expression import ExpressionScorer

        if (artifact.get("format") != 1 or artifact.get("passed") is not True
                or artifact.get("markers") != list(self.cfg.markers)
                or artifact.get("crop") != self.cfg.crop):
            raise ValueError("Expression artifact must pass validation and match the marker list and input size")
        self.critic = IHCCritic(2 * len(self.cfg.markers)).to(self.thresholds.device)
        self.expression_info = {key: artifact[key] for key in
                                ("format", "passed", "markers", "crop", "scope", "manifest_digest", "report")}
        for domain in ("HE", "IHC"):
            scorer = ExpressionScorer(len(self.cfg.markers)).to(self.thresholds.device)
            if "models" in artifact:
                scorer.load_state_dict(artifact["models"][domain])
            self.expression_scorers[domain] = scorer.eval().requires_grad_(False)

    def he_condition(self, image, tissue, marker):
        if self.expression_info is not None:
            score = self.expression_scorers["HE"](image, marker).sigmoid()
            # A single score describes the complete labeled unit, never its individual pixels.
            return score[:, None, None, None]
        he_h, default_tissue = he_projection(image, self.cfg)
        return self.probability(he_h / self.cfg.h_scale, default_tissue if tissue is None else tissue, marker)

    def probability(self, h, tissue, marker):
        _, logits = self.region(h, tissue, marker)
        return logits.float().sigmoid()

    def velocity(self, t, z, image, probability, marker):
        features = torch.cat((z, image, broadcast_grid(probability, image)), 1)
        return self.net(t, features, y=marker).float()

    def losses(self, h, reference, tissue, marker, image, he):
        region_loss = image.new_zeros(())
        if self.expression_info is None:
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
            torch.randn_like(image), image
        )
        # The target image is deliberately absent from the conditioning channels.
        # This is an unconditional IHC velocity objective, not a paired HE objective.
        prob = image.new_zeros((image.shape[0], 1, 1, 1)) if self.expression_info else torch.zeros_like(logits)
        prediction = self.velocity(t, z, torch.zeros_like(image), prob, marker)
        flow_loss = F.mse_loss(prediction, target_velocity)
        he_prob = self.he_condition(he, None, marker).detach()
        # Unroll from noise to an endpoint conditioned on an independent HE crop.
        state = torch.randn_like(he)
        steps = self.cfg.ode_steps if self.expression_info else self.cfg.nce_steps
        for i in range(steps):
            step_t = torch.full((he.shape[0],), i / steps, device=he.device)
            state = state + self.velocity(step_t, state, he, he_prob, marker) / steps
        nce = patch_nce(self.patch_encoder, he, state)
        return region_loss, flow_loss, nce, state.clamp(-1, 1)


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
            nn.Conv2d(10, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 24, 3, padding=1), nn.SiLU(),
            nn.Conv2d(24, 3, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def observation(self, h, z, velocity, probability):
        coarse = F.avg_pool2d(torch.cat((h, z, velocity), 1), self.region)
        return torch.cat((coarse, probability.expand(-1, -1, *coarse.shape[-2:])), 1).float()

    def forward(self, state):
        return self.net(state).float().clamp(-5, 5)


def he_projection(image, cfg):
    rgb = ((image.float() + 1) / 2).clamp(0, 1)
    od = -torch.log((rgb * 255 + 1) / 256)
    h = F.normalize(od.new_tensor(cfg.stain_h), dim=0)
    e = F.normalize(od.new_tensor(cfg.stain_e), dim=0)
    vector = h - (h @ e) * e
    hmap = (od * vector[None,:,None,None]).sum(1, keepdim=True) / (vector @ h)
    return hmap.clamp_min(0), od.amax(1, keepdim=True) >= cfg.tissue_od


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
