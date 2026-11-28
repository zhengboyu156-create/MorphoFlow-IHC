# Flow 40-epoch official MIST HER2 test

## Provenance and protocol

- Tested artifact: Flow-stage checkpoint, step 185,680 (40 epochs x 4,642 training pairs). PPO was not included.
- Only the model state dict was transferred from the remote host; optimizer state, logs, cache, and PPO artifacts were not transferred.
- Model weights SHA-256: `255de60aa986aa49903c18db9a8494d651bc3b0c9b29132829d73a99f854c109` (remote and local match).
- Code branch at evaluation: `fast/endpoint-cfm-4step-ppo-20261128`, commit `72bb450586e1afa78f3d58bca1ba57f298f46ef9`.
- Config SHA-256: `f0fd285fcb84e85737a6e008815f78c269eb3a456579978f6a315597f3d94cf0`; it matches the remote training config.
- Official test split: MIST HER2 `valA/valB`, 1,000 pairs. All 2,000 test image files matched manifest byte counts and SHA-256; all 1,000 pairs had unique IDs and finite metrics.
- Local environment: PyTorch 2.7.1+cu128, CUDA enabled, NVIDIA RTX 5070 Ti.
- Inference: native full-image, FP32, four-step Euler, fixed seed 1234 per image, no PPO policy. Total evaluation time: 534.3 seconds.

## Results

Per-image regional support-fraction MAE (lower is better; range 0-1):

| Metric | Mean | Median | SD |
|---|---:|---:|---:|
| 32 px regions | 0.3553 | 0.3527 | 0.0471 |
| 128 px reward regions | 0.2913 | 0.2940 | 0.0714 |

The 128 px value is the scale used by the PPO regional reward. These are regional HER2 DAB support-fraction errors; they are not pixel-wise registration metrics, PSNR/SSIM, or evidence of cell-level correspondence. This evaluates the Flow-only model before PPO; it does not assess whether PPO improves results.

## Artifacts

- `per_image_metrics.csv`: all 1,000 paired results.
- `summary.json`: aggregate metrics and hashes.
- `evaluate_flow40.py`: reproducible local evaluator.
- The model weight file is intentionally excluded from Git.
