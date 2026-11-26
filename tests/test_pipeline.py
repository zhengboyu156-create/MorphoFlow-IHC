import csv
import json
from dataclasses import asdict

import numpy as np
from PIL import Image
import pytest
import torch

from morphoflow_ihc.cli import main
from morphoflow_ihc.config import Config
from morphoflow_ihc.data import IHCCrops, _manifest_rows, prepare
from morphoflow_ihc.model import VelocityPolicy
from morphoflow_ihc.morphology import LocalMorphologyFPN, hematoxylin_map, morphology_channels
from morphoflow_ihc.stains import decompose, unit
from morphoflow_ihc.trajectory import action_distribution, policy_update, terminal_reward


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
    observation = torch.rand(1, 12, 2, 2)
    with torch.no_grad():
        mean = model(observation)
        actor = action_distribution(mean, cfg.action_std)
        actions = actor.sample((2,))
        experience = observation, mean, actions, actor.log_prob(actions), torch.tensor([[0.9], [0.2]])
    result = policy_update(model, optimizer, experience, cfg, passes=1)
    assert result["accepted"] == 1
    assert result["kl"] > 0
    assert result["signal_fraction"] == 1
    assert not torch.equal(mean, model(observation))
    tied = (*experience[:-1], torch.tensor([[0.5000001], [0.5]]))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    result = policy_update(model, optimizer, tied, cfg, passes=1)
    assert result["accepted"] == 0
    assert result["signal_fraction"] == 0
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())

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
    manifest.write_text("path,marker,group,domain,split,pair_id\n"
                        "a.png,HER2,p1,IHC,train,p1-a\n"
                        "b.png,,p1,HE,train,p1-a\n"
                        "c.png,HER2,p1,IHC,test,p1-b\n"
                        "d.png,,p1,HE,test,p1-b\n")
    with pytest.raises(ValueError, match="Source group spans splits"):
        prepare(manifest, tmp_path / "bad-cache", Config())


def test_unmatched_pair_is_rejected_before_opening_images(tmp_path):
    manifest = tmp_path / "images.csv"
    manifest.write_text("path,marker,group,domain,split,pair_id\n"
                        "missing.png,HER2,p1,IHC,train,pair-1\n")
    with pytest.raises(ValueError, match="exactly one H&E and one IHC"):
        prepare(manifest, tmp_path / "bad-cache", Config())


def test_mist_archive_pairs_by_source_prefix_and_splits_whole_prefixes(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = ["split,source_prefix,filename,bytes,sha256"]
    for group in ("case-a", "case-b", "case-c"):
        for split in ("trainA", "trainB", "valA", "valB"):
            section = "train" if split.startswith("train") else "val"
            rows.append(f"{split},{group},{group}_{section}_1_1.jpg,0,hash")
    manifest.write_text("\n".join(rows) + "\n")
    normalized = _manifest_rows(manifest, Config(markers={"HER2": 0.35}))
    grouped = {}
    modalities = {}
    for row in normalized:
        grouped.setdefault(row["group"], set()).add(row["split"])
        modalities.setdefault(row["pair_id"], set()).add(row["domain"])
    assert len(normalized) == 12
    assert all(len(splits) == 1 for splits in grouped.values())
    assert all(domains == {"HE", "IHC"} for domains in modalities.values())


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
        output.writerow(("path", "marker", "group", "domain", "split", "pair_id"))
        output.writerows((("HER2.png", "HER2", "p1", "IHC", "train", "pair-her2"),
                          ("he.png", "", "p1", "HE", "train", "pair-her2"),
                          ("KI67.png", "KI67", "p2", "IHC", "train", "pair-ki67"),
                          ("he.png", "", "p2", "HE", "train", "pair-ki67"),
                          ("unopened.png", "HER2", "p3", "IHC", "test", "pair-test"),
                          ("unopened-he.png", "", "p3", "HE", "test", "pair-test")))
    cache, flow, rl = [str(tmp_path / x) for x in ("cache", "flow.pt", "rl.pt")]
    cfg_path = str(config)
    main(["prepare", "--manifest", str(manifest), "--config", cfg_path, "--output", cache])
    paired = IHCCrops(tmp_path / "cache", cfg)
    marker, paired_ihc, paired_he = paired[0]
    assert marker in (0, 1)
    assert paired_ihc.shape == paired_he.shape == (3, 32, 32)
    base = ["train", "--cache", cache, "--config", cfg_path, "--output", flow, "--device", "cpu"]
    main(base + ["--steps", "1", "--batch", "2"])
    main(base + ["--steps", "2", "--resume", flow])
    main(["train-rl", "--cache", cache, "--checkpoint", flow, "--output", rl, "--groups", "2", "--device", "cpu"])
    joint = torch.load(rl, weights_only=True)
    initial = torch.load(flow, weights_only=True)
    assert joint["step"] == 2
    assert {"flow_optimizer", "critic_optimizer"} <= joint.keys()
    assert any(not torch.equal(value, initial["model"][name])
               for name, value in joint["model"].items() if name.startswith("net."))
    main(["train-rl", "--cache", cache, "--checkpoint", rl, "--output", rl,
          "--groups", "3", "--device", "cpu"])
    assert torch.load(rl, weights_only=True)["step"] == 3
    direct = tmp_path / "direct.pt"
    main(["train-rl", "--cache", cache, "--checkpoint", flow, "--output", str(direct),
          "--objective", "direct", "--groups", "2", "--device", "cpu"])
    assert torch.load(direct, weights_only=True)["policy_objective"] == "direct"
    out, maps, evaluation = [str(tmp_path / x) for x in ("virtual.png", "maps.npz", "score.json")]
    main(["infer", "--checkpoint", rl, "--image", str(source), "--marker", "KI67", "--output", out,
          "--maps", maps, "--device", "cpu"])
    assert Image.open(out).size == (41, 39)
    with np.load(maps) as data:
        assert np.isfinite(data["dab"]).all()
        assert data.files == ["dab"]
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
    rgb = torch.rand(1, 3, 32, 32) * 2 - 1
    he = torch.rand(1, 3, 32, 32) * 2 - 1
    _, nce, _ = model.losses(rgb, he, torch.tensor([0]))
    nce.backward()
    assert torch.isfinite(nce) and nce.item() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.net.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.patch_encoder.parameters())
    old = tmp_path / 'old.pt'
    torch.save({'format': 1, 'stage': 'flow'}, old)
    with pytest.raises(ValueError, match='checkpoint'):
        open_checkpoint(old, torch.device('cpu'))
    legacy = tmp_path / 'legacy-format-8.pt'
    legacy_config = asdict(cfg) | {'advantage_scale': 0.01}
    torch.save({'format': 8, 'stage': 'flow', 'config': legacy_config,
                'model': model.state_dict()}, legacy)
    reopened, policy, _ = open_checkpoint(legacy, torch.device('cpu'))
    assert reopened.cfg.min_reward_spread == cfg.min_reward_spread
    assert policy is None


def test_rgb_dab_diagnostic_matches_numpy_decomposition():
    from morphoflow_ihc.trajectory import rgb_to_dab
    cfg = Config()
    rng = np.random.default_rng(5)
    rgb = rng.integers(0, 256, size=(17, 19, 3), dtype=np.uint8)
    _, expected, _ = decompose(rgb, cfg, 'IHC')
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 127.5 - 1
    actual = rgb_to_dab(x, cfg)[0, 0].numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-5)


def test_morphology_projection_cancels_pure_dab_change():
    cfg = Config()
    h = torch.linspace(0.05, 0.8, 64).reshape(1, 1, 8, 8)
    h_vector = torch.tensor(cfg.stain_h)
    h_vector = h_vector / h_vector.norm()
    d_vector = torch.tensor(cfg.stain_d)
    d_vector = d_vector / d_vector.norm()
    images = []
    for dab in (0.1, 0.9):
        od = h[:, None, 0] * h_vector[None, :, None, None] + dab * d_vector[None, :, None, None]
        images.append((2 * torch.exp(-od) - 1).clamp(-1, 1))
    first = hematoxylin_map(images[0], cfg, "IHC")
    second = hematoxylin_map(images[1], cfg, "IHC")
    # The RGB<->OD convention includes the standard +1/256 stabilizer, so the
    # cancellation is numerical rather than bit exact.
    assert (first - second).abs().max() < 1e-3


def test_local_morphology_fpn_is_dense_and_soft_not_a_hard_segmentation():
    cfg = Config(crop=32, region=8, context=16, overlap=8)
    image = torch.rand(2, 3, 32, 32) * 2 - 1
    structure = morphology_channels(image, cfg, "HE")
    features, compartments = LocalMorphologyFPN(16, 24)(structure)
    assert features.shape == (2, 24, 8, 8)
    assert compartments.shape == (2, 3, 8, 8)
    torch.testing.assert_close(compartments.sum(1), torch.ones(2, 8, 8), atol=1e-6, rtol=0)
    assert ((compartments > 0) & (compartments < 1)).all()


def test_flow_target_uses_its_paired_he_condition():
    from morphoflow_ihc.model import StainFlow
    torch.set_num_threads(2)
    cfg = Config(markers={'HER2': 0.35}, crop=32, region=8, context=16, overlap=8,
                 channel_mult=(1, 1), ode_steps=4, action_step=2, nce_weight=0)
    model = StainFlow(cfg)
    real_velocity = model.velocity
    observed = []
    def traced(t, z, condition, marker):
        observed.append(condition.detach().clone())
        return real_velocity(t, z, condition, marker)
    model.velocity = traced
    rgb = torch.rand(2, 3, 32, 32) * 2 - 1
    he = torch.rand_like(rgb) * 2 - 1
    model.losses(rgb, he, torch.zeros(2, dtype=torch.long))
    assert len(observed) == cfg.ode_steps + 1
    for condition in observed:
        torch.testing.assert_close(condition, he)


def test_paired_reward_prefers_reference_and_stays_bounded():
    from morphoflow_ihc.model import StainFlow
    from morphoflow_ihc.trajectory import paired_expression_reward
    torch.manual_seed(12)
    cfg = Config(markers={'HER2': 0.35}, crop=32, region=8, context=16, overlap=8,
                 channel_mult=(1, 1), ode_steps=4, action_step=2)
    model = StainFlow(cfg).eval()
    he_vector = torch.tensor(cfg.stain_h)
    he_vector = he_vector / he_vector.norm()
    dab_vector = torch.tensor(cfg.stain_d)
    dab_vector = dab_vector / dab_vector.norm()
    he_od = torch.full((1, 1, 32, 32), 0.2)
    dab = torch.zeros_like(he_od)
    dab[:, :, 4:12, 4:12] = 0.8
    dab[:, :, 20:28, 20:28] = 0.6
    reference = (2 * torch.exp(-he_od * he_vector[None, :, None, None]
                               - dab * dab_vector[None, :, None, None]) - 1)
    fake = (2 * torch.exp(-he_od * he_vector[None, :, None, None]) - 1)
    with torch.no_grad():
        good = paired_expression_reward(model, reference, reference, torch.zeros(1, dtype=torch.long))
        other = paired_expression_reward(model, fake, reference, torch.zeros(1, dtype=torch.long))
    assert torch.isfinite(good).all() and ((good >= 0) & (good <= 1)).all()
    assert torch.isfinite(other).all() and ((other >= 0) & (other <= 1)).all()
    assert torch.all(good > other)


def test_policy_action_can_modify_rgb_channels_independently():
    from morphoflow_ihc.trajectory import corrected_velocity
    cfg = Config(markers={'HER2': 0.35}, crop=32, region=8, context=16, overlap=8,
                 channel_mult=(1, 1), ode_steps=4, action_step=2)
    policy = VelocityPolicy(cfg)
    observation = torch.zeros(1, 12, 4, 4)
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


def test_paired_reward_gradient_reaches_flow():
    from morphoflow_ihc.model import StainFlow
    from morphoflow_ihc.trajectory import paired_expression_reward, sample_rgb

    torch.set_num_threads(2)
    torch.manual_seed(39)
    cfg = Config(crop=32, region=8, context=16, overlap=8,
                 channel_mult=(1, 1), ode_steps=2, action_step=0)
    model = StainFlow(cfg)
    model.critic.requires_grad_(False)
    model.patch_encoder.requires_grad_(False)
    policy = VelocityPolicy(cfg).requires_grad_(False)
    he = torch.rand(1, 3, 32, 32).mul(2).sub(1)
    marker = torch.zeros(1, dtype=torch.long)
    reference = torch.rand_like(he).mul(2).sub(1)
    generated = sample_rgb(model, he, marker, torch.randn_like(he), policy)
    reward = paired_expression_reward(model, generated, reference, marker).mean()
    reward.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
               for p in model.net.parameters())
    assert all(p.grad is None for module in (policy, model.critic, model.patch_encoder)
               for p in module.parameters())




def test_policy_guides_generator_rollout_without_being_updated_by_generator_step():
    from morphoflow_ihc.engine import flow_optimizers, flow_update
    from morphoflow_ihc.model import StainFlow

    torch.set_num_threads(2)
    torch.manual_seed(51)
    cfg = Config(markers={"HER2": 0.35}, crop=32, region=8, context=16, overlap=8,
                 channel_mult=(1, 1), ode_steps=2, action_step=0,
                 morphology_width=8, morphology_channels=8,
                 patch_projection_dim=8, nce_patches=4)
    model, policy = StainFlow(cfg), VelocityPolicy(cfg)
    with torch.no_grad():
        policy.net[-1].bias.fill_(0.5)
    image = torch.rand(1, 3, 32, 32).mul(2).sub(1)
    he = torch.rand_like(image).mul(2).sub(1)
    marker = torch.zeros(1, dtype=torch.long)
    torch.manual_seed(77)
    _, _, base = model.losses(image, he, marker)
    torch.manual_seed(77)
    _, _, guided = model.losses(image, he, marker, policy=policy)
    assert not torch.allclose(base, guided)

    generator_optimizer, critic_optimizer = flow_optimizers(model)
    generator_parameter = next(model.net.parameters())
    generator_before = generator_parameter.detach().clone()
    policy_before = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    flow_update(model, generator_optimizer, critic_optimizer, image, he, marker, policy=policy)
    assert not torch.equal(generator_before, generator_parameter)
    assert all(torch.equal(value, policy.state_dict()[name]) for name, value in policy_before.items())
    assert all(parameter.requires_grad for parameter in policy.parameters())


def test_cut_style_patchnce_uses_shared_positive_positions_and_gradients():
    from morphoflow_ihc.patchnce import PatchNCELoss, PatchSampleF

    torch.manual_seed(9)
    sampler = PatchSampleF(8, 16)
    source = torch.randn(2, 8, 4, 4)
    generated = torch.randn(2, 8, 4, 4, requires_grad=True)
    positions = torch.tensor([0, 2, 5, 9, 12, 15])
    key = sampler(source, positions)
    query = sampler(generated, positions)
    loss = PatchNCELoss(temperature=0.07)(query, key)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    assert generated.grad is not None and generated.grad.abs().sum() > 0
    assert all(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in sampler.parameters())
    assert key.shape == query.shape == (2, 6, 16)
