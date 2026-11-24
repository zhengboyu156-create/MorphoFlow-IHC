"""Configuration shared by preparation, training, and inference."""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path


@dataclass
class Config:
    markers: dict[str, float] = field(default_factory=lambda: {"HER2": 0.35, "KI67": 0.35})
    pixel_size_um: float = 0.4661
    crop: int = 256
    region: int = 32
    context: int = 128
    overlap: int = 64
    channels: int = 32
    channel_mult: tuple[int, ...] = (1, 2, 2, 4)
    residual_blocks: int = 1
    ode_steps: int = 8
    action_step: int = 4
    nce_steps: int = 2
    nce_weight: float = 0.1
    tissue_od: float = 0.10
    stain_h: tuple[float, ...] = (0.650, 0.704, 0.286)
    stain_e: tuple[float, ...] = (0.072, 0.990, 0.105)
    stain_d: tuple[float, ...] = (0.268, 0.570, 0.776)
    h_scale: float = 1.0
    intensity_weight: float = 0.20
    action_std: float = 0.20
    velocity_limit: float = 0.10
    ppo_clip: float = 0.15
    kl_limit: float = 0.02
    flow_lr: float = 0.0002
    rl_lr: float = 0.00001

    def __post_init__(self):
        canonical = {key.upper(): float(value) for key, value in self.markers.items()}
        if not canonical or len(canonical) != len(self.markers):
            raise ValueError("Provide nonempty unique marker names")
        self.markers = dict(sorted(canonical.items()))
        self.channel_mult = tuple(self.channel_mult)
        if not self.channel_mult or min(self.channel_mult) < 1 or self.channels % 32:
            raise ValueError("U-Net channels must be divisible by 32 and its widths must be positive")
        if self.channels < 32 or self.residual_blocks < 1:
            raise ValueError("Invalid U-Net width or depth")
        if self.region < 4 or self.crop < 16 or self.crop % self.region:
            raise ValueError("crop must be divisible by region (at least 4 pixels)")
        if self.crop % (2 ** (len(self.channel_mult) - 1)):
            raise ValueError("crop is incompatible with U-Net downsampling")
        if self.context <= self.region or self.context % self.region:
            raise ValueError("context must span multiple region cells")
        if not 0 <= self.overlap < self.crop:
            raise ValueError("overlap must be smaller than crop")
        if self.nce_steps < 1 or not math.isfinite(self.nce_weight) or self.nce_weight <= 0:
            raise ValueError("PatchNCE settings must be positive")
        if not 0 <= self.action_step < self.ode_steps or self.ode_steps < 2:
            raise ValueError("action_step is zero-based and must index the Euler grid")
        positive = (self.pixel_size_um, self.h_scale, self.tissue_od,
                    self.action_std, self.velocity_limit, self.kl_limit, self.flow_lr, self.rl_lr,
                    *self.markers.values())
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Physical and optimization scales must be finite and positive")
        if not 0 <= self.intensity_weight <= 1 or not 0 < self.ppo_clip < 1:
            raise ValueError("Invalid reward or PPO weight")
        for name in ("stain_h", "stain_e", "stain_d"):
            vector = tuple(getattr(self, name))
            if len(vector) != 3 or any(not math.isfinite(v) or v < 0 for v in vector) or not sum(vector):
                raise ValueError(f"Invalid optical-density vector: {name}")
            setattr(self, name, vector)

    def marker_id(self, name: str) -> int:
        key = name.upper()
        if key not in self.markers:
            raise ValueError(f"Unknown marker {name!r}. Configured markers: {', '.join(self.markers)}")
        return tuple(self.markers).index(key)


def read_config(path: Path) -> Config:
    return Config(**json.loads(path.read_text(encoding="utf-8")))
