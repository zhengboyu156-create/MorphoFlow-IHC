# Four-Candidate PPO Preflight (2026-11-27)

## Provenance

- Code commit tested: `6c6caf382a0713be4d4d9986ae287bde8525d5f7`
- Branch: `fix/overlap-tolerant-regional-reward-20261127`
- Runtime: Python 3.12.4, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 5070 Ti.
- PPO config: `configs/mist_her2.json`, 4 candidate trajectories, 8 groups, batch 1, seed 1234; same 10-step base flow checkpoint as prior smoke tests.
- Input cache, images, and model weights remain local; they are not included here.

## Checks

- Full pytest: 19 passed (`pytest.txt`).
- Four-candidate RTX PPO smoke: 6/8 groups had reward signal and updated; 2/8 were skipped by the unchanged low-signal gate. Raw group logs: `ppo_smoke.jsonl`.
- Legacy resume compatibility: loaded a prior stage-`rl` checkpoint without `trajectory_candidates`; effective count resolved to 2, as intended.
- Inference/evaluation smoke: one fixed validation H&E, same seed 1234, base and 4-candidate policy outputs. The evaluator returned both 32 px diagnostic and 128 px reward-scale support errors. Base: 0.4279 / 0.4267; policy: 0.4164 / 0.4148. Raw records: `validation_base.json`, `validation_policy.json`; sample path: `validation_subset.json`.

## Scope

This confirms tests, candidate sampling/training execution, legacy candidate-count fallback, and evaluation plumbing. Eight PPO groups and one validation pair do not establish policy quality or validate reward/error alignment. This is not formal training; the broader preflight remains incomplete.
