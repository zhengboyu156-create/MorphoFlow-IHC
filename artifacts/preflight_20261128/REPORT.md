# Local preflight results (2026-11-28)

## Code and environment

- Guide commit: c6caf63e90f303a10b13041d7444604a6feec1eb
- Implementation code baseline: 273f2f89b50eb09a18524edd484e7def94860b4b
- GPU: NVIDIA GeForce RTX 5070 Ti, 16,303 MiB
- Python 3.12.4, PyTorch 2.7.1+cu128
- Config, manifest and cache index SHA-256 values are recorded in results.json.

## Dataset cache

Rebuilt the format-4 paired RGB cache from the official MIST manifest. Preparation checked byte counts and SHA-256 for training images. The resulting index contains 4,642 training pairs and 1,000 official test pairs. Only the training pairs are present in cache entries. The official test cohort was not used in the smoke run or for model selection.

## Automated tests

python -m pytest -q: 28 passed.

## Flow smoke run

Ran 8 updates with seed 1234, batch size 1 and AMP auto, which resolved to BF16. Flow, PatchNCE, critic and adversarial losses were finite at steps 1 and 8. The checkpoint is format 10 at step 8. Endpoint losses and checkpoint provenance are recorded in results.json.

## PPO smoke run

Ran 8 groups using the same cache, seed, batch and effective precision. Every group had signal_fraction 1.0. Reward spread ranged from about 1.60e-5 to 8.11e-5. Each group completed two policy update passes, and the maximum recorded KL was about 6.98e-6. This verifies the update path, not improved image quality.

The checkpoint is format 10 at group 8 with objective ppo.

## Inference

Loaded the joint checkpoint and generated a prediction from one trainA image with policy enabled. The result is finite RGB uint8 at 1024×1024 and is saved locally at runs/preflight_20261128/train_sample.png.

## Result

The smoke run passed the code, cache, Flow/PPO update, checkpoint and policy-guided full-image inference checks. It meets the engineering preflight criterion for beginning the formal 40+10 run. It does not measure official test metrics or establish that any module improves pathology results.

The checkpoints and generated image remain local and are excluded from Git.


## Interpretation clarification

The phrase “meets the engineering preflight criterion for beginning the formal 40+10 run” refers only to code execution, cache integrity, checkpoint and inference plumbing. It is **not an efficacy gate**. The 8 Flow updates, 8 PPO groups and one training-image inference do not establish that regional CFM tolerates displacement, PatchNCE preserves morphology, PatchGAN improves IHC appearance, or PPO improves held-out regional staining. The efficacy status is **not yet tested**; follow `docs/experiments/PREFORMAL_EFFICACY_GUIDE_20261128.md` before making any such claim. The official 1,000-pair test split remains reserved for the locked final evaluation.
