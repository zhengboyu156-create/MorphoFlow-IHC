"""Prepare IHC-only training caches and sample native-scale tissue crops."""

import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .config import Config
from .stains import canonical_size, decompose


def atomic_write(path: Path, writer):
    """Replace a file atomically while preserving any previous checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(dir=path.parent, delete=False)
    try:
        with temporary:
            writer(temporary)
        Path(temporary.name).replace(path)
    finally:
        Path(temporary.name).unlink(missing_ok=True)


def cache_settings(cfg: Config) -> dict:
    return {
        "markers": list(cfg.markers),
        "pixel_size_um": cfg.pixel_size_um,
        "optical_density": 1,
        "stain_h": list(cfg.stain_h),
        "stain_d": list(cfg.stain_d),
        "tissue_od": cfg.tissue_od,
    }


def prepare(manifest: Path, output: Path, cfg: Config) -> int:
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = {"path", "marker", "group", "domain", "split"}
        if not fields.issubset(reader.fieldnames or ()):
            raise ValueError(f"CSV requires columns {', '.join(sorted(fields))}")
        rows = list(reader)

    groups = {}
    path_splits = {}
    selected = []
    he_selected = []
    for row in rows:
        split = row["split"].strip().lower()
        group = row["group"].strip()
        domain = row["domain"].strip().upper().replace("&", "")
        if not group or not row["path"].strip() or split not in {"train", "val", "test"}:
            raise ValueError("Each image needs a path, patient/slide group and a valid split")
        if domain not in {"HE", "IHC"}:
            raise ValueError("domain must be HE or IHC")
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Patient/slide group spans splits: {group}")
        source = (manifest.parent / row["path"].strip()).resolve()
        if path_splits.setdefault(source, split) != split:
            raise ValueError(f"Image path spans splits: {source}")
        if domain == "HE" and split == "train":
            he_selected.append((source, float(row.get("mpp") or cfg.pixel_size_um)))
        if domain == "IHC" and split == "train":
            marker = row["marker"].strip().upper()
            cfg.marker_id(marker)
            mpp = float(row.get("mpp") or cfg.pixel_size_um)
            selected.append((source, marker, mpp))

    if not selected or len({source for source, _, _ in selected}) != len(selected):
        raise ValueError("Expected a nonempty IHC training pool with unique image paths")
    missing_markers = set(cfg.markers).difference(entry[1] for entry in selected)
    if missing_markers:
        raise ValueError(f"No training IHC images for {', '.join(sorted(missing_markers))}")

    if not he_selected:
        raise ValueError("Expected at least one unpaired H&E training image")
    output.mkdir(parents=True, exist_ok=True)
    # A unique cache generation keeps an earlier index valid during preparation.
    payload_dir = Path(mkdtemp(prefix="samples-", dir=output))
    entries = []
    for index, (source, marker, mpp) in enumerate(selected):
        name = f"{index:06d}.npz"
        with Image.open(source) as raw:
            rgb = raw.convert("RGB")
            size = canonical_size(rgb.size, mpp, cfg)
            if size != rgb.size:
                rgb = rgb.resize(size, Image.Resampling.BILINEAR)
            h, d, tissue = decompose(np.asarray(rgb), cfg, "IHC")
        atomic_write(payload_dir / name, lambda f: np.savez(f, h=h, d=d, tissue=tissue, rgb=np.asarray(rgb)))
        entries.append({"file": f"{payload_dir.name}/{name}", "marker": marker})
    he_entries = []
    for index, (source, mpp) in enumerate(he_selected):
        with Image.open(source) as raw:
            rgb = raw.convert("RGB")
            size = canonical_size(rgb.size, mpp, cfg)
            if size != rgb.size:
                rgb = rgb.resize(size, Image.Resampling.BILINEAR)
            image = np.asarray(rgb)
        name = f"he-{index:06d}.npz"
        atomic_write(payload_dir / name, lambda f: np.savez(f, rgb=image))
        he_entries.append(f"{payload_dir.name}/{name}")
    cache = {"format": 2, "settings": cache_settings(cfg), "entries": entries, "he_entries": he_entries}
    atomic_write(output / "index.json", lambda f: f.write(json.dumps(cache).encode()))
    return len(entries)


class IHCCrops(Dataset):
    def __init__(self, path: Path, cfg: Config, prioritize_positive: bool = False):
        self.root = path
        self.cfg = cfg
        self.prioritize_positive = prioritize_positive
        index = json.loads((path / "index.json").read_text(encoding="utf-8"))
        if index["settings"] != cache_settings(cfg):
            raise ValueError("Cache stain vectors, markers or pixel spacing differ from this configuration")
        if index.get("format") != 2 or not index.get("he_entries"):
            raise ValueError("RGB PatchNCE requires a format 2 cache including unpaired H&E; rerun prepare")
        self.he_entries = index["he_entries"]
        self.entries = index["entries"]
        if not self.entries:
            raise ValueError("Empty training cache")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        item = self.entries[index]
        marker, size = item["marker"], self.cfg.crop
        with np.load(self.root / item["file"], allow_pickle=False) as source:
            h, d, tissue, rgb = (source[key] for key in ("h", "d", "tissue", "rgb"))
        height, width = h.shape
        y = int(torch.randint(max(1, height - size + 1), (1,)).item())
        x = int(torch.randint(max(1, width - size + 1), (1,)).item())
        if self.prioritize_positive and float(torch.rand(())) < 0.5:
            positives = np.flatnonzero(tissue & (d > self.cfg.markers[marker]))
            if len(positives):
                cy, cx = divmod(int(positives[torch.randint(len(positives), (1,)).item()]), width)
                y = max(0, min(cy - size // 2, height - size))
                x = max(0, min(cx - size // 2, width - size))
        pad = ((0, max(0, size - height)), (0, max(0, size - width)))
        crop = (slice(y, y + size), slice(x, x + size))
        values = [torch.from_numpy(np.pad(a[crop], pad).copy())[None] for a in (h, d, tissue)]
        rgb_crop = np.pad(rgb[crop], (pad[0], pad[1], (0, 0)), constant_values=255)
        image = torch.from_numpy(rgb_crop.copy()).permute(2, 0, 1).float() / 127.5 - 1
        he_file = self.he_entries[int(torch.randint(len(self.he_entries), (1,)).item())]
        with np.load(self.root / he_file, allow_pickle=False) as source:
            he = source["rgb"]
        hh, ww = he.shape[:2]
        yy = int(torch.randint(max(1, hh-size+1), (1,)).item())
        xx = int(torch.randint(max(1, ww-size+1), (1,)).item())
        he_crop = he[yy:yy+size, xx:xx+size]
        he_crop = np.pad(he_crop, ((0,max(0,size-hh)),(0,max(0,size-ww)),(0,0)), constant_values=255)
        he_tensor = torch.from_numpy(he_crop.copy()).permute(2,0,1).float() / 127.5 - 1
        return (*values, self.cfg.marker_id(marker), image, he_tensor)
