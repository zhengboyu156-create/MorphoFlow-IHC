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
from morphoflow_ihc.stains import decompose, unit
from morphoflow_ihc.trajectory import action_distribution, policy_update, terminal_reward


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
    observation = torch.rand(1, 10, 2, 2)
    with torch.no_grad():
        mean = model(observation)
        actor = action_distribution(mean, cfg.action_std)
        actions = actor.sample((2,))
        experience = observation, mean, actions, actor.log_prob(actions), torch.tensor([[0.9], [0.2]])
    result = policy_update(model, optimizer, experience, cfg, passes=1)
    assert result["accepted"] == 1
    assert result["kl"] > 0
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
        rgb_ihc = np.rint(256 * np.exp(-h[..., None] * unit(cfg.stain_h) - d[..., None] * unit(cfg.stain_d)) - 1)
        Image.fromarray(np.clip(rgb_ihc, 0, 255).astype(np.uint8)).save(tmp_path / f"{marker}.png")
    source = tmp_path / "he.png"
    rgb = np.rint(256 * np.exp(-h[..., None] * unit(cfg.stain_h) - .2 * unit(cfg.stain_e)) - 1)
    Image.fromarray(rgb.astype(np.uint8)).save(source)
    manifest = tmp_path / "images.csv"
    with manifest.open("w", newline="") as stream:
        output = csv.writer(stream)
        output.writerow(("path", "marker", "group", "domain", "split"))
        output.writerows((("HER2.png", "HER2", "p1", "IHC", "train"),
                          ("KI67.png", "KI67", "p2", "IHC", "train"),
                          ("he.png", "", "p4", "HE", "train"),
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
        assert "h" not in data
    main(["evaluate", "--prediction", maps, "--reference", str(tmp_path / "KI67.png"), "--marker",
          "KI67", "--config", cfg_path, "--output", evaluation])
    report = json.loads(open(evaluation).read())
    assert 0 <= report["region_support_fraction_mae"] <= 1
    assert "pixel_support_reward" not in report


def test_patchnce_reaches_rgb_generator_and_old_checkpoints_rejected(tmp_path):
    from morphoflow_ihc.model import StainFlow
    from morphoflow_ihc.engine import open_checkpoint
    torch.set_num_threads(2)
    cfg = Config(markers={"HER2": 0.35}, crop=32, region=8, context=16,
                 overlap=8, channel_mult=(1, 1), ode_steps=4, action_step=2)
    model = StainFlow(cfg)
    h = torch.rand(1, 1, 32, 32)
    dab = torch.rand(1, 1, 32, 32)
    tissue = torch.ones_like(h, dtype=torch.bool)
    rgb = torch.rand(1, 3, 32, 32) * 2 - 1
    he = torch.rand(1, 3, 32, 32) * 2 - 1
    _, _, nce, _ = model.losses(h, dab, tissue, torch.tensor([0]), rgb, he)
    nce.backward()
    assert torch.isfinite(nce) and nce.item() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.net.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.patch_encoder.parameters())
    old = tmp_path / 'old.pt'
    torch.save({'format': 1, 'stage': 'flow'}, old)
    with pytest.raises(ValueError, match='checkpoint'):
        open_checkpoint(old, torch.device('cpu'))


def test_rgb_dab_diagnostic_matches_numpy_decomposition():
    from morphoflow_ihc.trajectory import rgb_to_dab
    cfg = Config()
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, size=(17, 19, 3), dtype=np.uint8)
    _, expected, _ = decompose(rgb, cfg, 'IHC')
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
    actual = rgb_to_dab(x, cfg)[0, 0].numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5)


def test_flow_target_is_not_passed_as_condition():
    from morphoflow_ihc.model import StainFlow
    torch.set_num_threads(2)
    cfg = Config(markers={'HER2': 0.35}, crop=32, region=8, context=16, overlap=8,
                 channel_mult=(1, 1), ode_steps=4, action_step=2, nce_weight=0)
    model = StainFlow(cfg)
    real_velocity = model.velocity
    observed = []
    def traced(t, z, condition, probability, marker):
        observed.append((condition.detach().clone(), probability.detach().clone()))
        return real_velocity(t, z, condition, probability, marker)
    model.velocity = traced
    rgb = torch.rand(2, 3, 32, 32) * 2 - 1
    he = torch.rand_like(rgb) * 2 - 1
    h = torch.rand(2, 1, 32, 32)
    model.losses(h, h, torch.ones_like(h, dtype=torch.bool), torch.zeros(2, dtype=torch.long), rgb, he)
    assert len(observed) == cfg.nce_steps + 1
    assert torch.count_nonzero(observed[0][0]) == 0
    assert torch.count_nonzero(observed[0][1]) == 0
    torch.testing.assert_close(observed[1][0], he)


def test_unpaired_reward_depends_on_generated_image_and_stays_bounded():
    from morphoflow_ihc.model import StainFlow
    from morphoflow_ihc.trajectory import unpaired_reward
    torch.manual_seed(12)
    cfg = Config(markers={'HER2': 0.35}, crop=32, region=8, context=16, overlap=8,
                 channel_mult=(1, 1), ode_steps=4, action_step=2)
    model = StainFlow(cfg).eval()
    he = torch.randn(2, 3, 32, 32).clamp(-1, 1)
    fake = torch.randn_like(he).clamp(-1, 1)
    idx = torch.arange(16)
    with torch.no_grad():
        good = unpaired_reward(model, he, he, torch.zeros(2, dtype=torch.long), idx)
        other = unpaired_reward(model, he, fake, torch.zeros(2, dtype=torch.long), idx)
    assert torch.isfinite(good).all() and ((good >= 0) & (good <= 1)).all()
    assert torch.isfinite(other).all() and not torch.equal(good, other)


def test_policy_action_can_modify_rgb_channels_independently():
    from morphoflow_ihc.trajectory import corrected_velocity
    cfg = Config(markers={'HER2': 0.35}, crop=32, region=8, context=16, overlap=8,
                 channel_mult=(1, 1), ode_steps=4, action_step=2)
    policy = VelocityPolicy(cfg)
    observation = torch.zeros(1, 10, 4, 4)
    assert policy(observation).shape == (1, 3, 4, 4)
    class Dummy:
        pass
    model = Dummy()
    model.cfg = cfg
    base = torch.ones(1, 3, 32, 32)
    action = torch.zeros(1, 3, 4, 4)
    action[:, 0] = 2
    output = corrected_velocity(model, base, action)
    assert torch.all(output[:, 0] > 1)
    torch.testing.assert_close(output[:, 1:], base[:, 1:])
    assert torch.max(torch.abs(output - base)) <= cfg.velocity_limit + 1e-6
