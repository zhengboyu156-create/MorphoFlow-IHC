"""Label-free RGB flow and IHC distribution critic."""

import torch
from torch import nn
import torch.nn.functional as F
from torchcfm import ConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModel

from .config import Config
from .morphology import LocalMorphologyFPN, morphology_channels
from .patchnce import PatchSampleF, paired_patchnce


def regional_velocity_mse(prediction, target, window):
    """Four-phase regional CFM loss with valid-area weighting at image edges."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("CFM velocity tensors must have identical BCHW shapes")
    batch, channels, height, width = prediction.shape
    half = max(1, window // 2)
    losses = []
    for top, left in ((0, 0), (0, half), (half, 0), (half, half)):
        bottom = (-(height + top)) % window
        right = (-(width + left)) % window
        pad = (left, right, top, bottom)
        valid = F.pad(prediction.new_ones((1, 1, height, width), dtype=torch.float32), pad)
        counts = F.avg_pool2d(valid, window, stride=window) * (window * window)
        pred_sum = F.avg_pool2d(F.pad(prediction.float(), pad), window, stride=window) * (window * window)
        target_sum = F.avg_pool2d(F.pad(target.float(), pad), window, stride=window) * (window * window)
        pred_mean = pred_sum / counts.clamp_min(1)
        target_mean = target_sum / counts.clamp_min(1)
        squared = (pred_mean - target_mean).square()
        losses.append((squared * counts).sum() / (counts.sum() * batch * channels).clamp_min(1))
    return torch.stack(losses).mean()


def single_step_endpoint(state, time, velocity):
    """Legacy bridge endpoint helper; not used for generated-image losses."""
    time = time.reshape(-1, *([1] * (state.ndim - 1)))
    return (state + (1 - time) * velocity).clamp(-1, 1)


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
            # 512px crops exceed the 2080 Ti memory budget without recomputing
            # residual-block activations during backward.
            use_checkpoint=True,
        )
        self.patch_encoder = LocalMorphologyFPN(cfg.morphology_width, cfg.morphology_channels)
        self.patch_sampler = PatchSampleF(cfg.morphology_channels, cfg.patch_projection_dim)
        self.critic = IHCCritic(len(cfg.markers))
        self.matcher = ConditionalFlowMatcher(sigma=0.0)

    def velocity(self, t, z, image, marker):
        features = torch.cat((z, image), 1)
        return self.net(t, features, y=marker).float()

    def morphology(self, image, domain):
        return self.patch_encoder(morphology_channels(image, self.cfg, domain))

    def morphology_loss(self, he, generated):
        source, _ = self.morphology(he, "HE")
        output, _ = self.morphology(generated, "IHC")
        return paired_patchnce(source, output, self.patch_sampler,
                               patches=self.cfg.nce_patches,
                               temperature=self.cfg.nce_temperature)

    def losses(self, image, he, marker, policy=None):
        t, z, target_velocity = self.matcher.sample_location_and_conditional_flow(
            torch.randn_like(image), image
        )
        prediction = self.velocity(t, z, he, marker)
        # Pair crops can be locally misregistered. Supervise conditional
        # velocity at regional scale over four grid phases. This preserves
        # coarse spatial expression but does not imply cell-level matching.
        flow_loss = regional_velocity_mse(prediction, target_velocity, self.cfg.flow_region)

        # Use the same noise-to-endpoint solver path for auxiliary generator
        # losses that is used at inference. A CFM bridge state z contains part
        # of the paired IHC target; using it as the auxiliary fake would make
        # PatchNCE/PatchGAN see a different distribution from deployment.
        from .trajectory import sample_rgb
        generated = sample_rgb(self, he, marker, torch.randn_like(he), policy)
        return flow_loss, self.morphology_loss(he, generated), generated


class IHCCritic(nn.Module):
    """Marker-conditioned IHC-only PatchGAN for stain realism."""

    def __init__(self, markers: int, base_channels: int = 32, layers: int = 3):
        super().__init__()
        self.markers = markers
        input_channels = 3 + markers  # IHC RGB plus marker indicator map
        sequence = [nn.Conv2d(input_channels, base_channels, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)]
        nf_mult = 1
        for layer in range(1, layers):
            nf_mult_previous = nf_mult
            nf_mult = min(2 ** layer, 8)
            sequence.extend((nn.Conv2d(base_channels * nf_mult_previous, base_channels * nf_mult,
                                       4, stride=2, padding=1, bias=False),
                             nn.BatchNorm2d(base_channels * nf_mult),
                             nn.LeakyReLU(0.2, inplace=True)))
        nf_mult_previous = nf_mult
        nf_mult = min(2 ** layers, 8)
        sequence.extend((nn.Conv2d(base_channels * nf_mult_previous, base_channels * nf_mult,
                                   4, stride=1, padding=1, bias=False),
                         nn.BatchNorm2d(base_channels * nf_mult),
                         nn.LeakyReLU(0.2, inplace=True),
                         nn.Conv2d(base_channels * nf_mult, 1, 4, stride=1, padding=1)))
        self.model = nn.Sequential(*sequence)

    def forward(self, image, marker):
        marker_map = F.one_hot(marker, num_classes=self.markers).to(image.dtype)
        marker_map = marker_map[:, :, None, None].expand(-1, -1, *image.shape[-2:])
        return self.model(torch.cat((image.float(), marker_map.float()), dim=1))


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
