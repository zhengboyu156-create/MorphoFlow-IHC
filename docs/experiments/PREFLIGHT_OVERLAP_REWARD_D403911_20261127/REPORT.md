# Overlap-Tolerant Reward Preflight (2026-11-27)

## Provenance

- Code commit tested: `d4039116a53beeac76d8157edb5c36dbf1086fd3`
- Branch: `fix/overlap-tolerant-regional-reward-20261127`
- Runtime: Python 3.12.4, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 5070 Ti.
- Dataset: MIST HER2, 11,284 manifest rows; verified local source image hashes against manifest. Manifest SHA-256: `bd667a845ddf15071815c6f31a042d190abc353ecf2c14f5cfc5f6d8c6695571`.
- Patient independence is not claimed; split grouping remains `source_prefix`. Cache: 4,533 train / 600 val / 509 test pairs.
- No source images, predictions, model weights, or cache arrays are included here.

## Tests and smoke run

- `python -m pytest -q`: 18 passed. Raw output: `pytest.txt`.
- PPO: 8 groups, batch 1, seed 1234, starting from the fixed 10-step flow smoke checkpoint. Full records: `ppo_smoke.jsonl`.
- The recorded PPO run had 6/8 groups with informative reward signal; 2/8 groups were skipped by the low-spread guard. Informative groups accepted two PPO passes. The guard remains active.

## Fixed validation comparison

Five validation pairs were fixed before comparison. Base and PPO outputs used the same H&E, seed (1234), marker HER2, and the branch's overlapping regional evaluator (threshold 0.35; region 32 px). Across 10 predictions, reward versus negative regional support error had Spearman rho 0.879 (descriptive only). PPO reward increased on all five paired comparisons; region support error decreased on four and increased on one. Full values are in `reward_direction.json`; pair manifest is `validation_subset.json`.

## Structural displacement audit

The same five pairs were examined in hematoxylin optical-density channels with 256 px windows and 128 px stride (241 tissue-overlap windows). Median phase-correlation response was 0.017, and no window reached response 0.1. This exploratory method did not yield a reliable local displacement estimate. Detailed outputs are in `displacement_qc.json`. Overlapping reward windows reduce fixed-grid sensitivity; they do not correct large or nonrigid misregistration.

## Decision

**NO-GO for formal training.** The new regression test and full test suite pass, and reward/error direction is broadly consistent on this small subset. However, reward signal is absent for some PPO samples, one of five validation pairs worsened, and structural correspondence could not be estimated reliably. Expand validation and address/characterize section mismatch before formal training.
