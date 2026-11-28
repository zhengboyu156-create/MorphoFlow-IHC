"""Build deterministic, source-prefix-balanced MIST pilot train/dev caches."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import random
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_caches(source, output, dev_pairs, seed):
    source = Path(source)
    output = Path(output)
    source_index_path = source / "index.json"
    payload = json.loads(source_index_path.read_text(encoding="utf-8"))
    if payload.get("format") != 4:
        raise ValueError("Expected the verified format 4 MIST cache")
    entries = [entry for entry in payload["entries"]
               if entry.get("split") == "train"
               and entry.get("source_cohort", "official_train") == "official_train"]
    if len(entries) != 4642:
        raise ValueError(f"Expected 4,642 official training pairs, found {len(entries)}")
    groups = defaultdict(list)
    for entry in entries:
        groups[entry["pair_id"].split("/", 1)[0]].append(entry)
    if dev_pairs <= len(groups) or dev_pairs >= len(entries):
        raise ValueError("Dev pair count must exceed available prefixes and leave training pairs")

    rng = random.Random(seed)
    prefixes = sorted(groups)
    for values in groups.values():
        values.sort(key=lambda row: row["pair_id"])
        rng.shuffle(values)
    rng.shuffle(prefixes)
    selected = []
    depth = 0
    while len(selected) < dev_pairs:
        advanced = False
        for prefix in prefixes:
            if depth < len(groups[prefix]) and len(selected) < dev_pairs:
                selected.append(groups[prefix][depth])
                advanced = True
        if not advanced:
            raise ValueError("Could not fill the requested balanced dev sample")
        depth += 1
    dev_ids = {row["pair_id"] for row in selected}
    train = [row for row in entries if row["pair_id"] not in dev_ids]
    dev_counts = Counter(row["pair_id"].split("/", 1)[0] for row in selected)

    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing pilot output: {output}")
    output.mkdir(parents=True)
    for subset, subset_entries in (("pilot_train_cache", train), ("pilot_dev_cache", selected)):
        relative_files = {row[key] for row in subset_entries for key in ("he_file", "ihc_file")}
        for relative in relative_files:
            source_file = source / relative
            target_file = output / subset / relative
            target_file.parent.mkdir(parents=True, exist_ok=True)
            target_file.hardlink_to(source_file)

    cache_root_index = {
        "format": 4,
        "settings": payload["settings"],
        "manifest_sha256": payload.get("manifest_sha256"),
        "entries": [],
        "source_cohort_split_counts": {},
    }
    for name, rows in (("pilot_train_cache", train), ("pilot_dev_cache", selected)):
        subset_rows = [dict(row, split="train") for row in rows]
        index = dict(cache_root_index, entries=subset_rows,
                     source_cohort_split_counts={"official_train": {"train": len(rows), "val": 0, "test": 0}})
        index_path = output / name / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")

    split = {
        "seed": seed,
        "source_cache_index_sha256": sha256(source_index_path),
        "source_cache_manifest_sha256": payload.get("manifest_sha256"),
        "source_train_pair_count": len(entries),
        "source_prefix_count": len(groups),
        "grouping": "first path component of pair_id; source prefix only, not asserted patient identity",
        "pilot_dev_pair_count": len(selected),
        "pilot_dev_prefix_count": len(dev_counts),
        "pilot_dev_prefix_counts": dict(sorted(dev_counts.items())),
        "pilot_dev_pair_ids": sorted(dev_ids),
        "pilot_train_pair_count": len(train),
        "pilot_train_cache_index_sha256": sha256(output / "pilot_train_cache" / "index.json"),
        "pilot_dev_cache_index_sha256": sha256(output / "pilot_dev_cache" / "index.json"),
    }
    (output / "pilot_split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in split.items() if key != "pilot_dev_pair_ids"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-pairs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261129)
    args = parser.parse_args()
    make_caches(args.source_cache, args.output, args.dev_pairs, args.seed)


if __name__ == "__main__":
    main()
