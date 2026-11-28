from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from morphoflow_ihc.config import Config
from morphoflow_ihc.data import sha256, verify_manifest_image
from morphoflow_ihc.engine import CHECKPOINT_FORMAT


@pytest.fixture
def evaluator(monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "artifacts" / "flow40_official_test_20261128"
    monkeypatch.syspath_prepend(str(folder))
    import mist_evaluation

    return mist_evaluation


def test_manifest_rejects_same_size_replacement_or_missing_hash(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    record = {"bytes": "5", "sha256": sha256(path)}
    verify_manifest_image(path, record)
    path.write_bytes(b"other")
    with pytest.raises(ValueError, match="does not match"):
        verify_manifest_image(path, record)
    with pytest.raises(ValueError, match="invalid manifest SHA-256"):
        verify_manifest_image(path, {"bytes": "5", "sha256": ""})


def test_manifest_rejects_duplicate_modality_rows(evaluator, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("split,source_prefix,filename\nvalA,group,img.jpg\nvalB,group,img.jpg\n")
    assert len(evaluator.read_pairs(manifest, 1)) == 1
    manifest.write_text(manifest.read_text() + "valB,group,img.jpg\n")
    with pytest.raises(ValueError, match="Duplicate valB"):
        evaluator.read_pairs(manifest, 1)


def test_rl_eval_provenance_distinguishes_policy_ablation_and_legacy(evaluator, tmp_path):
    cfg = Config(markers={"HER2": 0.35})
    path = tmp_path / "joint.pt"
    state = {"format": CHECKPOINT_FORMAT, "stage": "rl", "step": 7, "config": asdict(cfg),
             "model": {}, "policy": {}, "policy_objective": "ppo"}
    torch.save(state, path)
    args = SimpleNamespace(weights=path, checkpoint_step=None, allow_historical_weights=False,
                           disable_policy=False)
    device = torch.device("cpu")
    _, _, enabled = evaluator.checkpoint_for_evaluation(args, cfg, device, load_model=False)
    args.disable_policy = True
    _, _, disabled = evaluator.checkpoint_for_evaluation(args, cfg, device, load_model=False)
    assert enabled["policy_enabled"] and not disabled["policy_enabled"]
    assert disabled["checkpoint_stage"] == "rl" and disabled["training_objective_verified"]
    state["format"] = 9
    torch.save(state, path)
    with pytest.raises(ValueError, match="format 10"):
        evaluator.checkpoint_for_evaluation(args, cfg, device, load_model=False)
    args.allow_historical_weights = True
    _, _, historical = evaluator.checkpoint_for_evaluation(args, cfg, device, load_model=False)
    assert not historical["training_objective_verified"]
