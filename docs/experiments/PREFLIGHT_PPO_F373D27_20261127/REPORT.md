# PPO Preflight Results (2026-11-27)

## Provenance

- Repository commit: `f373d27f1e0eb0d9cc064fef935307b6503dee60`
- Branch: `experiment/morphology-fpn-teacher-20261126`
- Dataset: MIST HER2 bundled manifest, 11,284 image rows; manifest SHA-256: bd667a845ddf15071815c6f31a042d190abc353ecf2c14f5cfc5f6d8c6695571
- Local data was matched against every manifest SHA-256 before cache creation. Patient independence is not claimed; split unit is `source_prefix`.
- Runtime: Python 3.12.4, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 5070 Ti.
- Config: `configs/mist_her2.json`; preflight cache has 4,533 train / 600 val / 509 test pairs.
- This archive intentionally excludes images, prediction media, checkpoints, and cache arrays.

## Results

- `python -m pytest -q`: 17 passed. Raw output: `pytest.txt`.
- PPO smoke: 8 groups, batch 1, seed 1234, starting from the 10-step flow smoke checkpoint. Raw step records: `ppo_smoke.jsonl`.
- Reward signal: 5/8 groups had `signal_fraction=1`; 3/8 groups skipped policy updates due to low reward spread. Every informative group accepted two PPO passes. Signal coverage remains limited.
- Fixed validation subset: 5 pairs (10 base/policy predictions), same H&E, seed, marker, threshold (0.35), region size (32 px), and paired region-support metric. Reward vs negative region-support error Spearman rho = 0.879 across these 10 predictions. Policy reward rose on all five pairs; support error fell on four and rose on one. This is a small descriptive check, not a formal validation claim. Per-prediction values: `reward_direction.json`; frozen subset: `validation_subset.json`.
- Structural displacement audit: hematoxylin OD channels, 256 px windows, 128 px stride, 241 tissue-overlap windows across the same five pairs. Median phase-correlation response = 0.017. No tile reached response 0.1, so no reliable local displacement distribution could be estimated. Full exploratory measurements: `displacement_qc.json`.

## Decision

**NO-GO for formal training.** The PPO low-signal gate is functioning, but 3/8 smoke groups lacked usable reward spread and local structural correspondence could not be estimated reliably. Expand validation and resolve/characterize registration before treating paired pixel-level objectives or PPO reward alignment as validated. Patient-level independence is recorded as unverified and is not a blocker for this preflight, per the project decision.
