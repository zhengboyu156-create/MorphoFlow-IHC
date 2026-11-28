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


def sha256(path):
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest_image(path, row):
    expected_hash = row.get("sha256", "").strip().lower()
    if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
        raise ValueError(f"Missing or invalid manifest SHA-256 for {path}")
    try:
        expected_size = int(row.get("bytes"))
    except (ValueError, TypeError) as error:
        raise ValueError(f"Missing or invalid manifest byte count for {path}") from error
    if path.stat().st_size != expected_size or sha256(path) != expected_hash:
        raise ValueError(f"Image does not match manifest bytes/SHA-256: {path}")


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


def _manifest_rows(manifest: Path, cfg: Config):
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        rows = list(reader)

    # The upstream MIST authors define 4,642 HER2 training pairs and 1,000
    # testing pairs. The MAPS-GAN archive calls the latter valA/valB, so keep
    # that cohort as test. MIST has no upstream validation cohort; do not move
    # official test pairs into training or invent a validation split here.
    # Prefixes are not used as patient identifiers or split groups.
    if {"split", "source_prefix", "filename"}.issubset(fields):
        if len(cfg.markers) != 1:
            raise ValueError("The bundled MIST HER2 manifest requires a single-marker config")
        valid_splits = {"trainA", "trainB", "valA", "valB"}
        pair_cohorts = {}
        for row in rows:
            source_split = row["split"].strip()
            if source_split not in valid_splits:
                raise ValueError(f"Unsupported MIST source split: {source_split}")
            group, name = (row[field].strip() for field in ("source_prefix", "filename"))
            if any(not part or part in {".", ".."} or any(c in part for c in "/\\:,") for part in (group, name)):
                raise ValueError(f"Invalid MIST manifest path: {group}/{name}")
            pair_id = f"{group}/{name}"
            pair_cohorts.setdefault(pair_id, set()).add(source_split)
        for pair_id, cohorts in pair_cohorts.items():
            if cohorts not in ({"trainA", "trainB"}, {"valA", "valB"}):
                raise ValueError(f"MIST pair {pair_id!r} must have matching A/B cohort rows")

        normalized = []
        for row in rows:
            source_split = row["split"].strip()
            group = row["source_prefix"].strip()
            domain = "HE" if source_split.endswith("A") else "IHC"
            pair_id = f"{group}/{row['filename'].strip()}"
            is_official_test = source_split.startswith("val")
            normalized.append({
                "path": str(Path(source_split) / group / row["filename"].strip()),
                "marker": next(iter(cfg.markers)) if domain == "IHC" else "",
                # Pair-level grouping preserves the upstream cohort boundary;
                # source_prefix values occur in both upstream cohorts.
                "group": pair_id,
                "domain": domain,
                "split": "test" if is_official_test else "train",
                "pair_id": pair_id,
                "source_cohort": "official_test" if is_official_test else "official_train",
                "bytes": row.get("bytes"),
                "sha256": row.get("sha256") or "",
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
        source_cohort = row.get("source_cohort") or None
        entry = pairs.setdefault(pair_id, {"group": group, "split": split,
                                            "source_cohort": source_cohort, "modalities": {}})
        if entry["group"] != group or entry["split"] != split or domain in entry["modalities"]:
            raise ValueError(f"Pair {pair_id!r} has inconsistent group/split or duplicate modality")
        if entry["source_cohort"] != source_cohort:
            raise ValueError(f"Pair {pair_id!r} crosses source cohorts")
        mpp = float(row.get("mpp") or cfg.pixel_size_um)
        entry["modalities"][domain] = (source, mpp, row)
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
            source, mpp, row = entry["modalities"][domain]
            if entry["source_cohort"] == "official_train":
                verify_manifest_image(source, row)
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
        cache_entry = {"he_file": paths["HE"], "ihc_file": paths["IHC"],
                       "marker": entry["marker"], "pair_id": pair_id, "group": entry["group"],
                       "split": entry["split"]}
        if entry["source_cohort"] is not None:
            cache_entry["source_cohort"] = entry["source_cohort"]
        entries.append(cache_entry)
    source_cohort_split_counts = {}
    for entry in pairs.values():
        cohort = entry["source_cohort"]
        if cohort is None:
            continue
        counts = source_cohort_split_counts.setdefault(cohort, {split: 0 for split in ("train", "val", "test")})
        counts[entry["split"]] += 1
    cache = {"format": 4, "settings": cache_settings(cfg), "entries": entries, "manifest_sha256": sha256(manifest),
             "split_assignments": dict(sorted(groups.items())), "split_counts": split_counts}
    if source_cohort_split_counts:
        cache["source_cohort_split_counts"] = source_cohort_split_counts
    atomic_write(output / "index.json", lambda f: f.write(json.dumps(cache).encode()))
    return split_counts["train"]


class IHCCrops(Dataset):
    def __init__(self, path: Path, cfg: Config):
        self.root = path
        self.cfg = cfg
        payload = (path / "index.json").read_bytes()
        self.fingerprint = hashlib.sha256(payload).hexdigest()
        index = json.loads(payload)
        if index["settings"] != cache_settings(cfg):
            raise ValueError("Cache stain vectors, markers or pixel spacing differ from this configuration")
        if index.get("format") != 4:
            raise ValueError("Expected a format 4 paired RGB cache; rerun prepare with paired data")
        self.entries = index["entries"]
        if not self.entries:
            raise ValueError("Empty training cache")
        if any(entry.get("split") != "train" for entry in self.entries):
            raise ValueError("Training cache contains non-training entries")
        if len({entry["pair_id"] for entry in self.entries}) != len(self.entries):
            raise ValueError("Training cache contains duplicate pair IDs")
        if index.get("source_cohort_split_counts") and not index.get("manifest_sha256"):
            raise ValueError("Rebuild the official cache to verify training source hashes")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        generator = None
        if isinstance(index, tuple):
            index, seed = index
            generator = torch.Generator().manual_seed(seed)
        item = self.entries[index]
        marker, size = item["marker"], self.cfg.crop
        ihc, he = self.paired_crop(item["ihc_file"], item["he_file"], size, generator)
        return self.cfg.marker_id(marker), ihc, he

    def paired_crop(self, ihc_file, he_file, size, generator=None):
        with np.load(self.root / ihc_file, allow_pickle=False) as source:
            ihc = source["rgb"]
        with np.load(self.root / he_file, allow_pickle=False) as source:
            he = source["rgb"]
        if ihc.shape != he.shape:
            raise ValueError("Cached paired H&E/IHC images differ in shape")
        height, width = ihc.shape[:2]
        y = int(torch.randint(max(1, height - size + 1), (1,), generator=generator).item())
        x = int(torch.randint(max(1, width - size + 1), (1,), generator=generator).item())
        def tensor_crop(rgb):
            patch = rgb[y:y + size, x:x + size]
            padding = ((0, size - patch.shape[0]), (0, size - patch.shape[1]), (0, 0))
            patch = np.pad(patch, padding, constant_values=255)
            return torch.from_numpy(patch).permute(2, 0, 1).float() / 127.5 - 1
        return tensor_crop(ihc), tensor_crop(he)
