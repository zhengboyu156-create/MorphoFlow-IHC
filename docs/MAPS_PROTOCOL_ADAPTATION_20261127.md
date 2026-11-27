# MAPS-aligned input and evaluation geometry

## Fixed geometry

| Stage | MorphoFlow-IHC setting | MAPS-GAN reference setting |
|---|---|---|
| Training input | Shared-coordinate random 512×512 H&E/IHC crops | 512×512 crops from native 1024×1024 images |
| Training batch | 1 | 1 |
| Evaluation input | Native 1024×1024 H&E | Native 1024×1024 full image |
| Evaluation inference | One full-image forward trajectory; no crop tiling/blending | Full-image evaluation |
| Dataset cohort | Preserve official MIST split: 4,642 train / 1,000 test | Same MIST cohort definition |

The MIST configuration and default Flow batch now use these geometry settings. PPO retains its own schedule and group-based update count; MAPS-GAN's 40+10 epoch schedule is not copied because MorphoFlow has a distinct two-stage Flow/PPO training procedure.

## Inference behavior

`morphoflow_ihc infer` converts the input to canonical pixel spacing when `--mpp` differs from the configured spacing, then evaluates the entire canonical image in a single trajectory and restores the original output dimensions. For the MAPS-comparable MIST evaluation, use the native 1024×1024 source images at the configured 0.4661 μm/pixel spacing, so no scaling is applied. The implementation no longer uses `crop` or `overlap` to tile inference.

Full-image inference has higher peak memory use than the previous 256 px tiled inference. The configuration change alone does not certify that 512 px batch-1 training or 1024 px inference fits every GPU. Existing CUDA preflight results were obtained with the prior 256 px crop/tiled inference and must not be cited as memory validation for the new geometry.

## Run provenance boundary

The official-split run recorded in `docs/experiments/FORMAL_TRAINING_MIST_HER2_OFFICIAL_SPLIT_20261127/` was launched with the earlier 256 px crop and batch 1. Its process has already loaded that configuration; this commit does not alter the live process, its cache, or its checkpoint. Results from that run remain 256-crop results. To use the MAPS-aligned geometry, start a **fresh run** from this branch and record its source commit, config hash, cache hash, and GPU memory/throughput. Do not resume the 256-crop checkpoint as if it had been trained with 512 px crops.

The official test cohort remains held out: use the 4,642 training pairs for fitting and evaluate all 1,000 official test pairs only after the training choices are fixed. The split correction changes the earlier custom 80/10/10 prefix-repartition experiment; do not mix results across those protocols.
