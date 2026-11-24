"""Whole-image expression supervision for independent H&E and IHC pools."""

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .config import read_config
from .data import atomic_write


@dataclass(frozen=True)
class LabeledImage:
    path: Path
    marker: int
    label: float


def manifest_digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read_pools(path, cfg):
    pools = {(domain, split, marker): [] for domain in ("HE", "IHC")
             for split in ("train", "val") for marker in range(len(cfg.markers))}
    groups, paths = {}, {}
    scope, endpoints = {}, {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "marker", "group", "domain", "split", "label", "label_source",
                    "expression_scope", "expression_endpoint"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Expression manifest missing columns: " + ", ".join(sorted(required - set(reader.fieldnames or ()))))
        for raw in reader:
            row = {k: (raw[k] or "").strip() for k in required}
            domain = row["domain"].upper().replace("&", "")
            split, group = row["split"].lower(), row["group"]
            if domain not in {"HE", "IHC"} or split not in {"train", "val", "test"} or not group or not row["path"]:
                raise ValueError("Each expression row requires domain, split, group and image path")
            source = (path.parent / row["path"]).resolve()
            if paths.setdefault(source, (split, domain, group)) != (split, domain, group):
                raise ValueError("Image identity crosses expression groups, domains or splits")
            if groups.setdefault(group, split) != split:
                raise ValueError("Expression groups must be split-disjoint")
            if split == "test":
                continue
            marker = cfg.marker_id(row["marker"])
            if row["expression_scope"] not in {"slide", "region"} or not row["expression_endpoint"]:
                raise ValueError("Declare slide/region scope and the binary expression endpoint")
            if scope.setdefault(marker, row["expression_scope"]) != row["expression_scope"]:
                raise ValueError("Use a single expression scope per marker")
            if endpoints.setdefault(marker, row["expression_endpoint"]) != row["expression_endpoint"]:
                raise ValueError("H&E and IHC must use the same expression endpoint")
            expected = "independent_assay" if domain == "HE" else "ihc_annotation"
            if row["label_source"] != expected or row["label"] not in {"0", "1"}:
                raise ValueError(f"{domain} requires a binary label with source {expected}")
            pools[(domain, split, marker)].append((group, LabeledImage(source, marker, float(row["label"]))))

    for split in ("train", "val"):
        he = {g for (d, s, _), records in pools.items() if (d, s) == ("HE", split) for g, _ in records}
        ihc = {g for (d, s, _), records in pools.items() if (d, s) == ("IHC", split) for g, _ in records}
        if he & ihc:
            raise ValueError("Expression H&E and IHC pools must use independent groups")
    for key, records in pools.items():
        if len({item.path for _, item in records}) != len(records):
            raise ValueError(f"Duplicate expression images in {key}")
        if {item.label for _, item in records} != {0.0, 1.0}:
            raise ValueError(f"Expression pool {key} must include both label classes")
        pools[key] = [item for _, item in records]
    return pools, {name: {"scope": scope[i], "endpoint": endpoints[i]}
                   for i, name in enumerate(cfg.markers)}


def whole_image(image, size):
    """Use the entire labeled unit, with aspect-ratio-preserving padding."""
    image = image.convert("RGB")
    factor = size / max(image.size)
    width, height = (max(1, round(axis * factor)) for axis in image.size)
    payload = np.full((size, size, 3), 255, dtype=np.uint8)
    payload[:height, :width] = np.asarray(image.resize((width, height), Image.Resampling.BILINEAR))
    tensor = torch.from_numpy(payload).permute(2, 0, 1).float() / 127.5 - 1
    return tensor, (width, height)


def load_unit(item, size):
    with Image.open(item.path) as image:
        return whole_image(image, size)[0]


class ExpressionScorer(nn.Module):
    def __init__(self, markers):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.output = nn.Linear(64, markers)

    def forward(self, image, marker):
        return self.output(self.encoder(image.float())).gather(1, marker[:, None]).squeeze(1).float()


def fit_expression(args):
    from .engine import choose_device

    cfg = read_config(args.config)
    pools, scope = read_pools(args.manifest, cfg)
    device = choose_device(args.device)
    torch.manual_seed(args.seed)
    state, report = {}, {}
    for domain in ("HE", "IHC"):
        scorer = ExpressionScorer(len(cfg.markers)).to(device)
        optimizer = torch.optim.AdamW(scorer.parameters(), lr=cfg.flow_lr)
        train = [item for i in range(len(cfg.markers)) for item in pools[(domain, "train", i)]]
        for _ in range(args.epochs):
            for indices in torch.randperm(len(train)).split(args.batch):
                records = [train[i] for i in indices.tolist()]
                image = torch.stack([load_unit(item, cfg.crop) for item in records]).to(device)
                marker = torch.tensor([item.marker for item in records], device=device)
                labels = torch.tensor([item.label for item in records], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(scorer(image, marker), labels)
                loss.backward()
                optimizer.step()
        scorer.eval()
        report[domain] = {}
        for i, name in enumerate(cfg.markers):
            records = pools[(domain, "val", i)]
            labels = torch.tensor([item.label for item in records])
            with torch.no_grad():
                predicted = torch.cat([
                    scorer(torch.stack([load_unit(records[j], cfg.crop) for j in range(start, end)]).to(device),
                           torch.full((end - start,), i, device=device)).sigmoid().cpu()
                    for start in range(0, len(records), args.batch)
                    for end in [min(start + args.batch, len(records))]
                ])
            prior = sum(item.label for item in pools[(domain, "train", i)]) / len(pools[(domain, "train", i)])
            brier = (predicted - labels).square().mean().item()
            baseline = (labels - prior).square().mean().item()
            ranking = (predicted[labels == 1].mean() > predicted[labels == 0].mean()).item()
            report[domain][name] = {"n": len(records), "brier": brier, "prior_brier": baseline,
                                    "positive_mean_above_negative": ranking,
                                    "passed": bool(brier < baseline and ranking)}
        state[domain] = scorer.state_dict()
    passed = all(row["passed"] for domain in report.values() for row in domain.values())
    artifact = {"format": 1, "markers": list(cfg.markers), "crop": cfg.crop, "scope": scope,
                "manifest_digest": manifest_digest(args.manifest), "report": report, "passed": passed,
                "models": state}
    atomic_write(args.output, lambda stream: torch.save(artifact, stream))
    print(json.dumps({"passed": passed, "validation": report}))
