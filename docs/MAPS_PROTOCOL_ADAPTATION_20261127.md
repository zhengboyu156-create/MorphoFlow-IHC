# MAPS-aligned input and evaluation geometry

## Fixed geometry

| Stage | MorphoFlow-IHC setting | MAPS-GAN reference setting |
|---|---|---|
| Training input | Shared-coordinate random 512×512 H&E/IHC crops | 512×512 crops from native 1024×1024 images |
| Training batch | 1 | 1 |
| Evaluation input | Native 1024×1024 H&E | Native 1024×1024 full image |
| Evaluation inference | One full-image forward trajectory; no crop tiling/blending | Full-image evaluation |
| Dataset cohort | Preserve official MIST split: 4,642 train / 1,000 test | Same MIST cohort definition |

The MIST configuration and default Flow batch use these geometry settings. The MAPS-GAN repository's current original-V3 preset specifies 40 constant-learning-rate epochs followed by 10 linearly decayed epochs. This experiment adapts that schedule to MorphoFlow's two stages as **40 Flow epochs + 10 PPO epochs**: Flow uses the base Flow/critic learning rates for its 40 passes; the 10 PPO passes update the policy, Flow, and critic together while all three learning rates decay linearly. With the official batch-1 train set, the targets are 185,680 Flow steps and 46,420 PPO groups.

This is an explicit schedule adaptation, not a claim that the two methods have identical optimization cost. Each MorphoFlow PPO group samples four candidate trajectories and performs an additional joint Flow update, so its compute per paired sample differs from MAPS-GAN's generator/discriminator update. The previous 25-epoch Flow + 3,000-group PPO plan did **not** follow the reference schedule and is superseded.

The MAPS-style LR factor stays at 1.0 through epoch 40, then follows the standard linear rule `1 - (decay_epoch / 11)` for the 10 decay epochs (10/11 down to 1/11 on epochs 41–50; the next epoch would be 0). The schedule is computed from completed global data passes, so resuming a checkpoint preserves its epoch position.

Reference: [MAPS-GAN active V3 preset](https://github.com/zhengboyu156-create/MAPS-GAN-CGA-BES-CDM/blob/main/configs/active_method.json), [V3 protocol](https://github.com/zhengboyu156-create/MAPS-GAN-CGA-BES-CDM/blob/main/docs/ACTIVE_V3.md), and [MIST/BCI comparison protocol](https://github.com/zhengboyu156-create/MAPS-GAN-CGA-BES-CDM/blob/main/docs/experiments/BCI_MIST_HER2_BASELINE_PROTOCOL.md).

## Inference behavior

`morphoflow_ihc infer` converts the input to canonical pixel spacing when `--mpp` differs from the configured spacing, then evaluates the entire canonical image in a single trajectory and restores the original output dimensions. For the MAPS-comparable MIST evaluation, use the native 1024×1024 source images at the configured 0.4661 μm/pixel spacing, so no scaling is applied. The implementation no longer uses `crop` or `overlap` to tile inference.

Full-image inference has higher peak memory use than the previous 256 px tiled inference. The configuration change alone does not certify that 512 px batch-1 training or 1024 px inference fits every GPU. Existing CUDA preflight results were obtained with the prior 256 px crop/tiled inference and must not be cited as memory validation for the new geometry.

## Run provenance boundary

The official-split run recorded in `docs/experiments/FORMAL_TRAINING_MIST_HER2_OFFICIAL_SPLIT_20261127/` was launched with the earlier 256 px crop and batch 1. Its process has already loaded that configuration; this commit does not alter the live process, its cache, or its checkpoint. Results from that run remain 256-crop results. To use the MAPS-aligned geometry, start a **fresh run** from this branch and record its source commit, config hash, cache hash, and GPU memory/throughput. Do not resume the 256-crop checkpoint as if it had been trained with 512 px crops.

The official test cohort remains held out: use the 4,642 training pairs for fitting and evaluate all 1,000 official test pairs only after the training choices are fixed. The split correction changes the earlier custom 80/10/10 prefix-repartition experiment; do not mix results across those protocols.
