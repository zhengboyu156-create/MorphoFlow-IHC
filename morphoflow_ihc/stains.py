"""Pixel-aligned optical-density decomposition and IHC rendering."""

import math

import numpy as np
from PIL import Image

from .config import Config


def optical_density(rgb: np.ndarray) -> np.ndarray:
    return -np.log((np.asarray(rgb, dtype=np.float64) + 1.0) / 256.0)


def unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def decompose(rgb: np.ndarray, cfg: Config, domain: str):
    """Recover H using a linear projector orthogonal to the other stain.

    DAB is estimated by two-component nonnegative least squares on IHC.
    The projector cancels only the specified dye vector in the linear model.
    """
    if domain not in {"HE", "IHC"}:
        raise ValueError(f"Unsupported stain domain: {domain}")
    od = optical_density(rgb)
    h = unit(cfg.stain_h)
    other = unit(cfg.stain_d if domain == "IHC" else cfg.stain_e)
    projection = h - (h @ other) * other
    denominator = projection @ h
    if denominator < 1e-5:
        raise ValueError("Stain vectors are nearly collinear")
    hematoxylin = np.maximum(od @ projection / denominator, 0).astype(np.float32)
    tissue = od.max(axis=-1) >= cfg.tissue_od
    if domain == "HE":
        return hematoxylin, None, tissue

    vectors = np.stack((h, other), axis=1)
    concentrations = od @ np.linalg.pinv(vectors).T
    boundary = (concentrations < 0).any(axis=-1)
    candidates = np.maximum(od[boundary] @ vectors, 0)
    edge = np.argmax(candidates, axis=1)
    concentrations[boundary] = np.eye(2)[edge] * candidates
    return hematoxylin, concentrations[..., 1].astype(np.float32), tissue


def canonical_size(size: tuple[int, int], source_mpp: float, cfg: Config) -> tuple[int, int]:
    if not math.isfinite(source_mpp) or source_mpp <= 0:
        raise ValueError("Pixel spacing must be finite and positive")
    return tuple(max(1, round(axis * source_mpp / cfg.pixel_size_um)) for axis in size)


def resize_map(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if size == array.shape[::-1]:
        return array
    return np.array(Image.fromarray(np.asarray(array, dtype=np.float32)).resize(size, Image.Resampling.BILINEAR))
