# MAPS-compatible MIST HER2 preflight (2026-11-27)

## Scope and provenance

- Geometry source commit: `2b8725367fbb56acae1d75d64a8caf3192d500db`; branch test-fixture correction: `bf897ff40a2f1726dd0107d6cece1aea61c06abd`.
- Official archive split: `trainA/trainB` = 4,642 paired training images; `valA/valB` = 1,000 paired images, treated as the official held-out test cohort. Training cache uses the official-split cache; training code does not read the held-out images.
- Cache index SHA-256: `b14d501b3c60fd4593e2e3c322965be50a4d9a4537aa5979bdf1c1576f1cb1e9`.
- Runtime: Python 3.10.8, PyTorch 2.1.2+cu118, NVIDIA GeForce RTX 2080 Ti (11,264 MiB). The installed PyTorch is below the repository's declared 2.7 minimum and is recorded as an environment deviation.
- Current code enables TorchCFM residual-block activation checkpointing to fit 512px training. The `direct` reward objective disables it because TorchCFM's custom checkpoint backward does not support the second-order gradient required by that objective. Default PPO keeps checkpointing enabled.

## Verification

- `python -m pytest -q`: **20 passed**, 1 upstream PyTorch TypedStorage deprecation warning, 11.37 s.
- CUDA Flow, 512x512 random-crop geometry, batch 1, 2 steps: completed 2/2; peak 6,242.1 MiB allocated / 6,836.0 MiB reserved.
- CUDA Flow, same geometry, 10 steps: completed 10/10 in 14.0 s including CLI/startup overhead (about 1.4 s/step in this short sample). Final logged losses: flow 0.177919, PatchNCE 4.21114, critic 0.596922, adversarial 0.723259.
- CUDA PPO smoke, 512px, batch 1, 4 candidates: 1-group case had reward spread 7.510e-6 below the 1e-5 gate and correctly skipped the policy update (`signal_fraction=0`). In a separate 8-group run, the logged first group also skipped (spread 9.269e-6); the final group had spread 1.335e-5, `signal_fraction=1`, and accepted the update. This is an implementation smoke, not evidence of stable reward quality; intermediate per-group statistics were not retained in the default log.
- CUDA full-image inference on a synthetic 1024x1024 input: completed; output remained 1024x1024; peak 2,739.2 MiB allocated / 2,928.0 MiB reserved.
- An earlier pre-checkpoint 512px step failed with CUDA OOM. Do not cite that failure as the current code result; checkpointing resolved the measured 512px Flow smoke. The preceding failed run is retained in the remote preflight log directory for provenance.

## Interpretation and release gate

Geometry, memory, code tests, and inference path now pass on this RTX 2080 Ti. The PPO reward signal is still close to the configured threshold and varies by sampled group; these short smokes do not establish that PPO improves validation performance. No official test image was used for this preflight, and no full training or test-set evaluation result is claimed here. Patient independence is excluded from this run's gating per the project decision; record it as a limitation.

The short throughput sample projects approximately 45.2 hours for 25 epochs at 4,642 pairs/epoch and 1.4 s/step. The configured Flow cap is 20 hours, so it is a time-capped run and may finish substantially fewer than 25 epochs. PPO's configured cap is 6 hours. These are estimates from a 10-step sample, not guarantees.

## Artifacts

Preflight logs, temporary checkpoints, and the synthetic input/output remain outside Git at:
`/root/autodl-tmp/morphoflow_maps512_20261127/runs/preflight_geom_20261127/`

Only source, this text report, and later text logs/metrics are intended for Git; model weights and images are excluded.