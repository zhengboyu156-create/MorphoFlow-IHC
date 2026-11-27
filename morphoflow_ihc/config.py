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
    reward_region: int | None = None
    flow_region: int = 64
    context: int = 128
    overlap: int = 64
    channels: int = 32
    channel_mult: tuple[int, ...] = (1, 2, 2, 4)
    residual_blocks: int = 1
    ode_steps: int = 8
    action_step: int = 4
    nce_weight: float = 0.1
    morphology_width: int = 32
    morphology_channels: int = 64
    patch_projection_dim: int = 128
    nce_patches: int = 64
    nce_temperature: float = 0.07
    adversarial_weight: float = 0.1
    critic_lr: float = 0.0002
    tissue_od: float = 0.10
    stain_h: tuple[float, ...] = (0.650, 0.704, 0.286)
    stain_e: tuple[float, ...] = (0.072, 0.990, 0.105)
    stain_d: tuple[float, ...] = (0.268, 0.570, 0.776)
    intensity_weight: float = 0.20
    action_std: float = 0.20
    velocity_limit: float = 0.10
    velocity_rms_floor: float = 0.05
    min_reward_spread: float = 1e-5
    trajectory_candidates: int = 4
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
        if self.reward_region is None:
            self.reward_region = min(128, self.crop)
        if (self.reward_region < self.region or self.reward_region > self.crop
                or self.crop % self.reward_region or self.reward_region % self.region):
            raise ValueError("reward_region must be a region-aligned divisor of crop, at least region")
        if (self.flow_region < self.region or self.flow_region > self.crop
                or self.crop % self.flow_region or self.flow_region % self.region):
            raise ValueError("flow_region must be a region-aligned divisor of crop, at least region")
        if self.crop % (2 ** (len(self.channel_mult) - 1)):
            raise ValueError("crop is incompatible with U-Net downsampling")
        if self.context <= self.region or self.context % self.region:
            raise ValueError("context must span multiple region cells")
        if not 0 <= self.overlap < self.crop:
            raise ValueError("overlap must be smaller than crop")
        if not math.isfinite(self.nce_weight) or self.nce_weight < 0:
            raise ValueError("PatchNCE weight must be finite and nonnegative")
        if self.morphology_width < 8 or self.morphology_channels < 8:
            raise ValueError("Morphology FPN widths must be at least 8")
        if self.patch_projection_dim < 8 or self.nce_patches < 2:
            raise ValueError("Patch projection and sample count must be at least 8 and 2")
        if not math.isfinite(self.nce_temperature) or self.nce_temperature <= 0:
            raise ValueError("PatchNCE temperature must be finite and positive")
        if not 0 <= self.action_step < self.ode_steps or self.ode_steps < 2:
            raise ValueError("action_step is zero-based and must index the Euler grid")
        positive = (self.pixel_size_um, self.tissue_od,
                    self.action_std, self.velocity_limit, self.velocity_rms_floor,
                    self.min_reward_spread,
                    self.kl_limit, self.flow_lr, self.rl_lr,
                    self.adversarial_weight, self.critic_lr,
                    *self.markers.values())
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Physical and optimization scales must be finite and positive")
        if (not isinstance(self.trajectory_candidates, int)
                or isinstance(self.trajectory_candidates, bool)
                or self.trajectory_candidates < 2):
            raise ValueError("trajectory_candidates must be an integer of at least two")
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
    values = json.loads(path.read_text(encoding="utf-8"))
    # Older configs exposed fixed PPO advantage scaling. The current update
    # normalizes within each candidate pair and uses min_reward_spread instead.
    values.pop("advantage_scale", None)
    return Config(**values)
