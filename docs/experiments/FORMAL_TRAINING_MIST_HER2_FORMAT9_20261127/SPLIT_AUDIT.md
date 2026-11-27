# MIST HER2 split audit — 2026-11-27

## Finding

The formal run did **not** use the MIST authors' published train/test split. The official authors' README reports HER2 as 4,642 training pairs and 1,000 testing pairs, with no separate validation split listed: [AdaptiveSupervisedPatchNCE — Downloading the MIST Dataset](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE#downloading-the-mist-dataset).

The bundled MAPS-GAN archive manifest names these original cohorts `trainA/trainB` and `valA/valB`; the latter 1,000 pairs correspond to the authors' test set. MorphoFlow-IHC's then-current loader ignored that cohort boundary and deterministically assigned all 5,642 pairs across 61 filename prefixes to a new 80/10/10 split.

## Reconstructed pair counts

| Original MIST cohort | Custom train | Custom validation | Custom test | Total |
|---|---:|---:|---:|---:|
| Official training (`trainA/trainB`) | 3,726 | 494 | 422 | 4,642 |
| Official test (`valA/valB` in this archive) | 807 | 106 | 87 | 1,000 |
| **Custom split total** | **4,533** | **600** | **509** | **5,642** |

Therefore 807 official test pairs entered the formal training cache. The resulting model cannot be reported as trained/evaluated under the official MIST split. The custom 509-pair holdout is a different split and must be labeled as such.

## Run disposition

The formal Flow run was stopped on 2026-11-27 at about 20:02 +08:00 after the issue was confirmed. The latest logged step was 8,000; `flow.pt` was preserved (about 49.7 MB, checkpoint mtime 20:00:26 +08:00). Its log and checkpoint are retained for provenance only and must not be presented as an official-split result. PPO was not started.

For an official-protocol rerun, keep all 1,000 official test pairs untouched. If a validation set is needed for decisions, carve it out of the 4,642 official training pairs and label it as a new validation subset; do not call it an official validation set. The existing checkpoint was trained with official test examples and must not be resumed for that rerun.

For the corrected official-protocol rerun, all 4,642 official training pairs are retained for training and all 1,000 official test pairs remain held out until final evaluation. MIST provides no separate official validation set. See `docs/experiments/FORMAL_TRAINING_MIST_HER2_OFFICIAL_SPLIT_20261127/RUN_PLAN.md`.
