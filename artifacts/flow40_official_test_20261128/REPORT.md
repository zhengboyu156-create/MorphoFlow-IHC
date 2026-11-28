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

## Standardized comparison metrics (added 2026-11-28)

The Flow-only checkpoint was evaluated using the same MIST HER2 comparison protocol used for the MAPS-GAN results. This is the official `TrainValAB/valA+valB` evaluation cohort: 1,000 paired images at native 1024×1024 RGB. Inference used FP32, four-step Euler sampling, per-image fixed noise seed 1234, with no resize/crop/flip and no PPO policy.

| Metric | Result | Aggregation |
|---|---:|---|
| PSNR ↑ | 12.1974 | Mean of per-image RGB PSNR, `data_range=255` |
| SSIM ↑ | 0.1289 | Mean of per-image RGB SSIM, `data_range=255` |
| FID ↓ | 335.0259 | `pytorch-fid`, InceptionV3 2048-dimensional features, batch size 4; generated images vs real `valB` |
| Rc ↑ | 0.0661 | Mean per-image Pearson correlation between grayscale generated IHC and source H&E |
| Rp ↑ | 0.1284 | Pearson correlation across images between real and generated whole-image Fiji H-DAB `Colour_2` IntDen |
| Rave ↑ | 0.0972 | `(Rc + Rp) / 2` |

Both Rc and Rp are weak on this run; the results do not support a claim that the Flow checkpoint preserves morphology or HER2 expression well. Rc is a global grayscale correlation, and Rp is a whole-image intensity correlation; neither proves cell-level correspondence or corrects section misregistration. This result evaluates the Flow stage only and says nothing about whether PPO improves it. The earlier 32/128 px regional support-fraction MAE remains a separate supplemental diagnostic, not a substitute for these comparison metrics.

The direct Fiji macro produced 1,000 unique rows for both real and generated images, each verified as 1024×1024, 8-bit, with exact expected IDs. Fiji returned exit code 1 for both runs; the result was accepted only because the full CSVs passed those checks. The complete per-image CSV includes PSNR, SSIM, Rc components, and the two Fiji IntDen values so Rp can be recomputed. No prediction images or model weights are included in Git.

- Metric summary: `comparison_metrics_summary.json`
- Per-image values and Fiji IntDen inputs: `per_image_comparison_metrics.csv`
- Reproducible evaluator and Fiji macro: `evaluate_comparison_metrics.py`, `direct_fiji_intden.ijm`
- The predictions were generated in 827.6 s. Standard metric/Fiji postprocessing reused those predictions and completed in 712.6 s.
