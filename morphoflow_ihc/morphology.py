"""Stain-neutral dense morphology features without explicit segmentation.

The local encoder follows the residual BasicBlock and top-down FPN design used
by torchvision.  It is implemented locally so the training core does not need
torchvision operators.  See docs/UPSTREAM.md for the pinned public reference.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def _unit(image: torch.Tensor, values) -> torch.Tensor:
    vector = image.new_tensor(values)
    return F.normalize(vector, dim=0)


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(8, channels), channels)


def hematoxylin_map(image: torch.Tensor, cfg, domain: str) -> torch.Tensor:
    """Differentiably estimate hematoxylin while cancelling the other dye.

    ``domain`` selects eosin for H&E and DAB for IHC.  DAB concentration is not
    returned and therefore cannot enter morphology matching through this API.
    """
    if domain not in {"HE", "IHC"}:
        raise ValueError(f"Unsupported morphology domain: {domain}")
    rgb = ((image.float() + 1) / 2).clamp(0, 1)
    optical_density = -torch.log((rgb * 255 + 1) / 256)
    h = _unit(image, cfg.stain_h)
    other = _unit(image, cfg.stain_e if domain == "HE" else cfg.stain_d)
    projection = h - torch.dot(h, other) * other
    denominator = torch.dot(projection, h)
    if denominator <= 1e-5:
        raise ValueError("Stain vectors are nearly collinear")
    return torch.einsum("c,bchw->bhw", projection, optical_density)[:, None].div(denominator).clamp_min(0)


def morphology_channels(image: torch.Tensor, cfg, domain: str) -> torch.Tensor:
    """Return H concentration, gradient magnitude and Laplacian channels."""
    h = hematoxylin_map(image, cfg, domain)
    dtype, device = h.dtype, h.device
    sobel_x = torch.tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)), dtype=dtype, device=device)[None, None] / 8
    sobel_y = sobel_x.transpose(-1, -2)
    laplace = torch.tensor(((0, 1, 0), (1, -4, 1), (0, 1, 0)), dtype=dtype, device=device)[None, None]
    gx = F.conv2d(h, sobel_x, padding=1)
    gy = F.conv2d(h, sobel_y, padding=1)
    gradient = torch.sqrt(gx.square() + gy.square() + 1e-8)
    curvature = F.conv2d(h, laplace, padding=1).abs()
    scale = h.flatten(2).mean(-1, keepdim=True).clamp_min(1e-3)[:, :, None]
    return torch.cat((h / scale, gradient / scale, curvature / scale), dim=1).clamp(0, 8)


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual projection when shape changes."""

    def __init__(self, input_channels: int, output_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False)
        self.norm2 = _norm(output_channels)
        self.skip = (nn.Identity() if stride == 1 and input_channels == output_channels else
                     nn.Sequential(nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                                   _norm(output_channels)))

    def forward(self, image):
        residual = self.skip(image)
        image = F.silu(self.norm1(self.conv1(image)))
        return F.silu(self.norm2(self.conv2(image)) + residual)


class LocalMorphologyFPN(nn.Module):
    """High-resolution residual pyramid and heuristic morphology weights."""

    def __init__(self, width: int = 32, output_channels: int = 64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 3, padding=1, bias=False),
                                  _norm(width), nn.SiLU())
        self.stage1 = BasicBlock(width, width)
        self.stage2 = BasicBlock(width, width * 2, stride=2)
        self.stage3 = BasicBlock(width * 2, width * 4, stride=2)
        self.lateral = nn.ModuleList((nn.Conv2d(width, output_channels, 1),
                                      nn.Conv2d(width * 2, output_channels, 1),
                                      nn.Conv2d(width * 4, output_channels, 1)))
        self.smooth = nn.ModuleList(nn.Conv2d(output_channels, output_channels, 3, padding=1)
                                    for _ in range(3))
        self.fuse = nn.Sequential(nn.Conv2d(3 * output_channels, output_channels, 1, bias=False),
                                  _norm(output_channels), nn.SiLU())

    def forward(self, structure: torch.Tensor):
        c1 = self.stage1(self.stem(structure))
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        p3 = self.lateral[2](c3)
        p2 = self.lateral[1](c2) + F.interpolate(p3, c2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral[0](c1) + F.interpolate(p2, c1.shape[-2:], mode="bilinear", align_corners=False)
        target = c3.shape[-2:]
        pyramid = [F.interpolate(layer(value), target, mode="bilinear", align_corners=False)
                   for layer, value in zip(self.smooth, (p1, p2, p3))]
        features = self.fuse(torch.cat(pyramid, dim=1))
        local = F.adaptive_avg_pool2d(structure[:, :2], target)
        hematoxylin_level, edge_level = local[:, :1], local[:, 1:2]
        residual_context = -(hematoxylin_level + edge_level)
        weights = torch.cat((hematoxylin_level, edge_level, residual_context), dim=1).softmax(1)
        return features, weights
