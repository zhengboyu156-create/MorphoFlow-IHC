import csv
import json

import numpy as np
from PIL import Image
import pytest
import torch

from morphoflow_ihc.cli import main
from morphoflow_ihc.config import Config
from morphoflow_ihc.data import prepare
from morphoflow_ihc.model import RegionPredictor, VelocityPolicy
from morphoflow_ihc.stains import decompose, render_od, render_rgb, unit
from morphoflow_ihc.trajectory import action_distribution, policy_update, terminal_reward


def test_renderer_keeps_the_counterstain():
    cfg = Config()
    h = np.array([[0.2, 0.7], [0.5, 0.1]])
    dab = np.array([[1.2, 0.0], [0.5, 1.4]])
    h_vector, d_vector = unit(cfg.stain_h), unit(cfg.stain_d)
    vector = h_vector - (h_vector @ d_vector) * d_vector
    recovered = render_od(h, dab, cfg) @ vector / (vector @ h_vector)
    np.testing.assert_allclose(h, recovered)


def test_context_branch_excludes_its_center():
    cfg = Config(crop=32, region=8, context=16, overlap=8)
    torch.manual_seed(99)
    model = RegionPredictor(cfg)
    image = torch.rand(1, 1, 32, 32)
    modified = image.clone()
    modified[:, :, 8:16, 8:16] = 1.5
    tissue = torch.ones_like(image, dtype=torch.bool)
    seed = torch.tensor([0])
    a, _ = model(image, tissue, seed)
    b, _ = model(modified, tissue, seed)
    torch.testing.assert_close(a[:, :, 1, 1], b[:, :, 1, 1], atol=2e-6, rtol=0)


def test_reward_rejects_uniform_typing_and_respects_absent_classes():
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, :4, :4] = 0.8
    tissue = torch.ones_like(target, dtype=torch.bool)
    threshold = torch.tensor([[[[0.35]]]])
    full = torch.ones_like(target)
    blank = torch.zeros_like(target)
    def score(prediction, truth=target):
        return terminal_reward(prediction, truth, tissue, threshold, 4, .2).item()
    assert score(target) == pytest.approx(1)
    assert score(full) == score(blank) == 0
    assert score(blank, blank) == 1


def test_policy_accepts_small_update_and_rolls_back_large_update():
    torch.manual_seed(12)
    cfg = Config(rl_lr=1e-4)
    model = VelocityPolicy(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.rl_lr)
    observation = torch.rand(1, 4, 2, 2)
    with torch.no_grad():
        mean = model(observation)
        actor = action_distribution(mean, cfg.action_std)
        actions = actor.sample((2,))
        experience = observation, mean, actions, actor.log_prob(actions), torch.tensor([[0.9], [0.2]])
    result = policy_update(model, optimizer, experience, cfg, passes=1)
    assert result["accepted"] == 1
    assert not torch.equal(mean, model(observation))
    tied = (*experience[:-1], torch.tensor([[0.5000001], [0.5]]))
    assert policy_update(model, optimizer, tied, cfg, passes=1)["accepted"] == 0

    cfg.kl_limit = 1e-14
    with torch.no_grad():
        current = model(observation)
        old = action_distribution(current, cfg.action_std)
        current_actions = old.sample((2,))
        experience = observation, current, current_actions, old.log_prob(current_actions), torch.tensor([[0.9], [0.2]])
    result = policy_update(model, optimizer, experience, cfg, passes=1)
    assert result["accepted"] == 0
    torch.testing.assert_close(model(observation), current, rtol=0, atol=0)


def test_group_leakage_detected_before_reading_any_image(tmp_path):
    manifest = tmp_path / "images.csv"
    manifest.write_text("path,marker,group,domain,split\na.png,HER2,p1,IHC,train\nb.png,,p1,HE,test\n")
    with pytest.raises(ValueError, match="spans splits"):
        prepare(manifest, tmp_path / "bad-cache", Config())


def test_cli_pipeline(tmp_path):
    torch.set_num_threads(2)
    config = tmp_path / "tiny.json"
    config.write_text(json.dumps({"crop": 32, "region": 8, "context": 16, "overlap": 8,
                                  "channel_mult": [1, 1], "ode_steps": 4, "action_step": 2}))
    cfg = Config(**json.loads(config.read_text()))
    yy, xx = np.mgrid[:39, :41]
    h = 0.2 + 0.6 * np.exp(-((yy - 20) ** 2 + (xx - 19) ** 2) / 48)
    d = ((yy - 20) ** 2 + (xx - 19) ** 2 < 36).astype(float) * 0.7
    for marker in ("HER2", "KI67"):
        render_rgb(h, d, cfg).save(tmp_path / f"{marker}.png")
    source = tmp_path / "he.png"
    rgb = np.rint(256 * np.exp(-h[..., None] * unit(cfg.stain_h) - .2 * unit(cfg.stain_e)) - 1)
    Image.fromarray(rgb.astype(np.uint8)).save(source)
    manifest = tmp_path / "images.csv"
    with manifest.open("w", newline="") as stream:
        output = csv.writer(stream)
        output.writerow(("path", "marker", "group", "domain", "split"))
        output.writerows((("HER2.png", "HER2", "p1", "IHC", "train"),
                          ("KI67.png", "KI67", "p2", "IHC", "train"),
                          ("unopened.png", "HER2", "p3", "IHC", "test")))
    cache, flow, rl = [str(tmp_path / x) for x in ("cache", "flow.pt", "rl.pt")]
    cfg_path = str(config)
    main(["prepare", "--manifest", str(manifest), "--config", cfg_path, "--output", cache])
    base = ["train", "--cache", cache, "--config", cfg_path, "--output", flow, "--device", "cpu"]
    main(base + ["--steps", "1", "--batch", "2"])
    main(base + ["--steps", "2", "--resume", flow])
    main(["train-rl", "--cache", cache, "--checkpoint", flow, "--output", rl, "--groups", "2", "--device", "cpu"])
    assert torch.load(rl, weights_only=True)["step"] == 2
    out, maps, evaluation = [str(tmp_path / x) for x in ("virtual.png", "maps.npz", "score.json")]
    main(["infer", "--checkpoint", rl, "--image", str(source), "--marker", "KI67", "--output", out,
          "--maps", maps, "--device", "cpu"])
    assert Image.open(out).size == (41, 39)
    with np.load(maps) as data:
        assert np.isfinite(data["dab"]).all()
        raw_h, _, _ = decompose(rgb.astype(np.uint8), cfg, "HE")
        np.testing.assert_allclose(data["h"], raw_h * cfg.counterstain_gain)
    main(["evaluate", "--prediction", maps, "--reference", str(tmp_path / "KI67.png"), "--marker",
          "KI67", "--config", cfg_path, "--output", evaluation])
    report = json.loads(open(evaluation).read())
    assert 0 <= report["region_support_fraction_mae"] <= 1
    assert "pixel_support_reward" not in report
