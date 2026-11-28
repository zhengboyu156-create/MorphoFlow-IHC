"""Paired, same-coordinate PatchNCE adapted from the official CUT design."""

import torch
from torch import nn
import torch.nn.functional as F


class PatchSampleF(nn.Module):
    """Project sampled positions from a single fused morphology feature map."""

    def __init__(self, input_channels: int, projection_channels: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, projection_channels),
            nn.ReLU(inplace=True),
            nn.Linear(projection_channels, projection_channels),
        )

    def forward(self, features: torch.Tensor, patch_ids: torch.Tensor):
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        if patch_ids.ndim != 1 or patch_ids.numel() == 0:
            raise ValueError("patch_ids must be a nonempty 1-D shared spatial index")
        if int(patch_ids.min()) < 0 or int(patch_ids.max()) >= height * width:
            raise ValueError("patch_ids are outside the morphology feature map")
        sampled = tokens.index_select(1, patch_ids.to(features.device))
        projected = self.mlp(sampled.reshape(batch * patch_ids.numel(), channels))
        projected = projected.reshape(batch, patch_ids.numel(), -1)
        return F.normalize(projected.float(), dim=-1)


class PatchNCELoss(nn.Module):
    """CUT-style same-image negatives with H&E/IHC same-position positives."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("PatchNCE temperature must be positive")
        self.temperature = temperature

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        if query.shape != key.shape or query.ndim != 3:
            raise ValueError("PatchNCE expects matching [batch, patches, channels] tensors")
        batch, patches, _ = query.shape
        if patches < 2:
            raise ValueError("PatchNCE needs at least two sampled positions")
        key = key.detach()
        positive = (query * key).sum(dim=-1, keepdim=True)
        negative = torch.bmm(query, key.transpose(1, 2))
        diagonal = torch.eye(patches, device=query.device, dtype=torch.bool)[None]
        negative = negative.masked_fill(diagonal, -10.0)
        logits = torch.cat((positive, negative), dim=-1) / self.temperature
        labels = torch.zeros(batch * patches, dtype=torch.long, device=query.device)
        return F.cross_entropy(logits.reshape(batch * patches, patches + 1), labels)


def paired_patchnce(source: torch.Tensor, generated: torch.Tensor, sampler: PatchSampleF,
                    patches: int = 64, temperature: float = 0.07) -> torch.Tensor:
    if source.shape != generated.shape or source.ndim != 4:
        raise ValueError("Paired morphology feature maps must have identical BCHW shapes")
    spatial = source.shape[-2] * source.shape[-1]
    count = min(patches, spatial)
    if count < 2:
        raise ValueError("PatchNCE needs at least two feature-map positions")
    patch_ids = torch.randperm(spatial, device=source.device)[:count]
    with torch.no_grad():
        key = sampler(source, patch_ids)
    query = sampler(generated, patch_ids)
    return PatchNCELoss(temperature)(query, key)
