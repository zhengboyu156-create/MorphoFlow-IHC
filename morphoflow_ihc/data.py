"""Cache paired RGB training images and sample shared-coordinate crops."""

import csv
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .config import Config
from .stains import canonical_size


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
        "pairing": "shared-coordinate-v1",
    }


def _group_split(prefixes):
    """Deterministically split source prefixes 80/10/10 without prefix leakage."""
    ordered = sorted(set(prefixes), key=lambda value: hashlib.sha256(
        ("MorphoFlow-IHC-paired-v1:" + value).encode("utf-8")).digest())
    if len(ordered) < 3:
        raise ValueError("At least three source groups are required for train/val/test splitting")
    n_test = max(1, round(len(ordered) * 0.1))
    n_val = max(1, round(len(ordered) * 0.1))
    n_train = len(ordered) - n_val - n_test
    return {group: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
            for i, group in enumerate(ordered)}


def _manifest_rows(manifest: Path, cfg: Config):
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        rows = list(reader)

    # The bundled MIST provenance manifest pairs A/B images by filename. Its
    # published crop split mixes source prefixes across train and validation,
    # so reassign whole prefixes before using it for model selection.
    if {"split", "source_prefix", "filename"}.issubset(fields):
        if len(cfg.markers) != 1:
            raise ValueError("The bundled MIST HER2 manifest requires a single-marker config")
        splits = _group_split(row["source_prefix"].strip() for row in rows)
        normalized = []
        for row in rows:
            source_split = row["split"].strip()
            if source_split not in {"trainA", "trainB", "valA", "valB"}:
                raise ValueError(f"Unsupported MIST source split: {source_split}")
            group = row["source_prefix"].strip()
            domain = "HE" if source_split.endswith("A") else "IHC"
            normalized.append({
                "path": str(Path(source_split) / group / row["filename"].strip()),
                "marker": next(iter(cfg.markers)) if domain == "IHC" else "",
                "group": group,
                "domain": domain,
                "split": splits[group],
                "pair_id": f"{group}/{row['filename'].strip()}",
                "mpp": row.get("mpp") or cfg.pixel_size_um,
            })
        return normalized

    required = {"path", "marker", "group", "domain", "split", "pair_id"}
    if not required.issubset(fields):
        raise ValueError("Paired manifest requires path, marker, group, domain, split, pair_id")
    return rows


def prepare(manifest: Path, output: Path, cfg: Config) -> int:
    groups = {}
    path_splits = {}
    pairs = {}
    rows = _manifest_rows(manifest, cfg)
    for row in rows:
        split = row["split"].strip().lower()
        group = row["group"].strip()
        pair_id = row["pair_id"].strip()
        domain = row["domain"].strip().upper().replace("&", "")
        if not group or not pair_id or not row["path"].strip() or split not in {"train", "val", "test"}:
            raise ValueError("Each image needs a path, pair id, group and valid split")
        if domain not in {"HE", "IHC"}:
            raise ValueError("domain must be HE or IHC")
        if groups.setdefault(group, split) != split:
            raise ValueError(f"Source group spans splits: {group}")
        source = (manifest.parent / row["path"].strip()).resolve()
        if path_splits.setdefault(source, split) != split:
            raise ValueError(f"Image path spans splits: {source}")
        entry = pairs.setdefault(pair_id, {"group": group, "split": split, "modalities": {}})
        if entry["group"] != group or entry["split"] != split or domain in entry["modalities"]:
            raise ValueError(f"Pair {pair_id!r} has inconsistent group/split or duplicate modality")
        mpp = float(row.get("mpp") or cfg.pixel_size_um)
        entry["modalities"][domain] = (source, mpp)
        if domain == "IHC":
            marker = row["marker"].strip().upper()
            cfg.marker_id(marker)
            entry["marker"] = marker

    if not pairs:
        raise ValueError("Expected a nonempty paired training set")
    for pair_id, entry in pairs.items():
        if set(entry["modalities"]) != {"HE", "IHC"} or "marker" not in entry:
            raise ValueError(f"Pair {pair_id!r} must contain exactly one H&E and one IHC image")
        if entry["modalities"]["HE"][0] == entry["modalities"]["IHC"][0]:
            raise ValueError(f"Pair {pair_id!r} reuses the same image for both modalities")
    selected = [entry for entry in pairs.values() if entry["split"] == "train"]
    missing_markers = set(cfg.markers).difference(entry["marker"] for entry in selected)
    if missing_markers:
        raise ValueError(f"No training IHC images for {', '.join(sorted(missing_markers))}")
    if not selected:
        raise ValueError("No paired training images after group split")
    output.mkdir(parents=True, exist_ok=True)
    # A unique cache generation keeps an earlier index valid during preparation.
    payload_dir = Path(mkdtemp(prefix="samples-", dir=output))
    entries = []
    split_counts = {split: sum(entry["split"] == split for entry in pairs.values())
                    for split in ("train", "val", "test")}
    for index, (pair_id, entry) in enumerate(sorted(pairs.items())):
        if entry["split"] != "train":
            continue
        paths = {}
        sizes = {}
        for domain in ("HE", "IHC"):
            source, mpp = entry["modalities"][domain]
            with Image.open(source) as raw:
                rgb = raw.convert("RGB")
                size = canonical_size(rgb.size, mpp, cfg)
                if size != rgb.size:
                    rgb = rgb.resize(size, Image.Resampling.BILINEAR)
                array = np.asarray(rgb)
            sizes[domain] = array.shape[:2]
            name = f"{index:06d}-{domain.lower()}.npz"
            atomic_write(payload_dir / name, lambda f, value=array: np.savez(f, rgb=value))
            paths[domain] = f"{payload_dir.name}/{name}"
        if sizes["HE"] != sizes["IHC"]:
            raise ValueError(f"Paired images must have equal canonical size: {pair_id}")
        entries.append({"he_file": paths["HE"], "ihc_file": paths["IHC"],
                        "marker": entry["marker"], "pair_id": pair_id, "group": entry["group"],
                        "split": entry["split"]})
    cache = {"format": 4, "settings": cache_settings(cfg), "entries": entries,
             "split_assignments": dict(sorted(groups.items())), "split_counts": split_counts}
    atomic_write(output / "index.json", lambda f: f.write(json.dumps(cache).encode()))
    return split_counts["train"]


class IHCCrops(Dataset):
    def __init__(self, path: Path, cfg: Config):
        self.root = path
        self.cfg = cfg
        index = json.loads((path / "index.json").read_text(encoding="utf-8"))
        if index["settings"] != cache_settings(cfg):
            raise ValueError("Cache stain vectors, markers or pixel spacing differ from this configuration")
        if index.get("format") != 4:
            raise ValueError("Expected a format 4 paired RGB cache; rerun prepare with paired data")
        self.entries = index["entries"]
        if not self.entries:
            raise ValueError("Empty training cache")
        self.marker_indices = {
            marker: [i for i, item in enumerate(self.entries) if item["marker"] == marker]
            for marker in self.cfg.markers
        }
        if any(not indices for indices in self.marker_indices.values()):
            raise ValueError("Every configured marker needs an IHC target pool")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        item = self.entries[index]
        marker, size = item["marker"], self.cfg.crop
        ihc, he = self.paired_crop(item["ihc_file"], item["he_file"], size)
        return self.cfg.marker_id(marker), ihc, he

    def sample_ihc(self, marker):
        """Draw an independently sampled IHC crop from the requested marker pool."""
        marker_name = tuple(self.cfg.markers)[int(marker)]
        candidates = self.marker_indices[marker_name]
        index = candidates[int(torch.randint(len(candidates), (1,)).item())]
        item = self.entries[index]
        with np.load(self.root / item["ihc_file"], allow_pickle=False) as source:
            rgb = source["rgb"]
        height, width = rgb.shape[:2]
        size = self.cfg.crop
        y = int(torch.randint(max(1, height - size + 1), (1,)).item())
        x = int(torch.randint(max(1, width - size + 1), (1,)).item())
        patch = rgb[y:y + size, x:x + size]
        padding = ((0, size - patch.shape[0]), (0, size - patch.shape[1]), (0, 0))
        patch = np.pad(patch, padding, constant_values=255)
        tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 127.5 - 1
        return tensor

    def paired_crop(self, ihc_file, he_file, size):
        with np.load(self.root / ihc_file, allow_pickle=False) as source:
            ihc = source["rgb"]
        with np.load(self.root / he_file, allow_pickle=False) as source:
            he = source["rgb"]
        if ihc.shape != he.shape:
            raise ValueError("Cached paired H&E/IHC images differ in shape")
        height, width = ihc.shape[:2]
        y = int(torch.randint(max(1, height - size + 1), (1,)).item())
        x = int(torch.randint(max(1, width - size + 1), (1,)).item())
        def tensor_crop(rgb):
            patch = rgb[y:y + size, x:x + size]
            padding = ((0, size - patch.shape[0]), (0, size - patch.shape[1]), (0, 0))
            patch = np.pad(patch, padding, constant_values=255)
            return torch.from_numpy(patch).permute(2, 0, 1).float() / 127.5 - 1
        return tensor_crop(ihc), tensor_crop(he)
