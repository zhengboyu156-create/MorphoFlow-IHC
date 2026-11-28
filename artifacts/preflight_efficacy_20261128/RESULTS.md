# MorphoFlow-IHC efficacy preflight — 2026-11-28

## Scope

This is a **training-path/control preflight**, not the efficacy pilot specified by `PREFORMAL_EFFICACY_GUIDE_20261128.md`. It confirms that the one-factor switches and PPO/zero-action paths execute on the local RTX GPU and emit finite losses and checkpoints. Each training arm was deliberately limited to 2 updates/groups. The 128-pair pilot-dev split was frozen but **not evaluated**. No official test images were read, and no full training was launched.

## Reproducibility

- Branch: `run/preformal-efficacy-20261128`; source baseline before these edits: `0440ae5`.
- Hardware/runtime: NVIDIA GeForce RTX 5070 Ti 16 GB, Python 3.12.4, PyTorch 2.7.1+cu128, CUDA 12.8, BF16 autocast.
- Pilot-train: 4,514 paired training images; pilot-dev: 128 pairs held out by `source_prefix` across two prefixes. This is **not patient-level independence**. Split definition and IDs are in [`pilot_split.json`](pilot_split.json); its SHA-256 is recorded in `measurements.json`.
- All six arms used pilot-train cache index SHA-256 `77596d0f54de3a17225af57334adeec6422588da2e0c28cc92097d3d92017e49`, seed `20261128`, batch 1, workers 0, BF16, and separate fresh outputs. The same regional baseline Flow checkpoint was used to start both RL arms.
- Commands were the same except for the named objective/ablation; logs are in `logs/`, configs in `configs/`. Checkpoint SHA-256 values are in `measurements.json`; weight files remain local and were not uploaded.

## Outcomes

| Arm | Updates | Evidence from final update | Preflight result |
|---|---:|---|---|
| Regional CFM baseline | 2 Flow | CFM `0.376340`; NCE `4.311775`; critic `0.689481`; adversarial `0.696240` | Completed with finite losses |
| Pixel CFM | 2 Flow | CFM `1.393177`; NCE `4.311841`; critic `0.689486`; adversarial `0.696185` | Switch changes the CFM objective; raw CFM values across objectives are not directly comparable as quality |
| PatchNCE off | 2 Flow | CFM `0.373342`; NCE `0`; critic `0.689473`; adversarial `0.696278` | NCE computation is skipped when its weight is zero |
| PatchGAN off | 2 Flow | CFM `0.376340`; NCE `4.311775`; critic `0`; adversarial `0` | Critic update is skipped; checkpoint critic optimizer has no state |
| PPO | 2 groups (2 generator updates) | reward spreads `3.58e-6`, `6.68e-6`; both below `min_reward_spread=1e-5`; `signal_fraction=0`, `accepted=0` for both | PPO path runs, but received no usable learning signal in this tiny run |
| Zero action | 2 groups (2 generator updates) | no rollout/update; finite Flow/critic losses | Control path runs from the same Flow checkpoint |

`python -m pytest -q`: **28 passed**. `git diff --check`: passed. New CLI switches were also inspected through `--help` before execution. Per-arm raw logs and hashes are retained; checkpoints and generated images are not committed.

## Interpretation and remaining work

The controls are operational, but **none of the four efficacy claims is validated**. Two updates cannot establish a useful training trajectory; the PPO result specifically shows that the current reward spread was too small to update the policy. There are no pilot-dev image metrics, no known-displacement CFM stress curve, no frozen morphology/stain metrics, and no confidence intervals. Do not use this report to claim module effectiveness or release the formal 40+10 run as efficacy-validated. The next meaningful preflight is the guide's matched 1,000-update screening plus its fixed pilot-dev evaluation, after any reward-signal diagnosis; the official 1,000-pair test cohort remains untouched.
