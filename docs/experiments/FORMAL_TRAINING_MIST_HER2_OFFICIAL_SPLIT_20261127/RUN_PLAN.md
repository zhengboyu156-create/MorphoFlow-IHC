# Formal MIST HER2 run — official split preserved

## Cohort and provenance

- Upstream MIST HER2 cohort: 4,642 official training pairs and 1,000 official test pairs; the authors do not list a separate validation set.
- The MAPS-GAN archive names the official test cohort `valA/valB`. The corrected loader maps it to `test`; no official test pair enters training.
- Expected cache counts: 4,642 train / 0 validation / 1,000 test. The cache payload contains only the 4,642 official training pairs, and `source_cohort_split_counts` must show `official_test` only under `test`.
- Source prefixes are not used as patient IDs or split groups. The official cohort boundary is preserved exactly.
- Source manifest SHA-256: `bd667a845ddf15071815c6f31a042d190abc353ecf2c14f5cfc5f6d8c6695571`.
- Source inventory: 11,284 image rows / 5,642 pairs, each previously checked remotely against manifest byte counts and SHA-256. Source tarball SHA-256: `22bc5d2d135db4a5572d2a3043592c00d4a94c74cce480b913d40a00f14bbe13`.
- Runtime: AutoDL RTX 2080 Ti, Python 3.10.8, PyTorch 2.1.2+cu118, below the repository-declared PyTorch 2.7 minimum.

## Fixed run settings

- Config `configs/mist_her2.json`, SHA-256 `e33a580361e04555922150e2779070135099bd356ed77c6f341c4b4ea9dc1c0d`.
- Seed 1234; batch size 1; workers 0.
- Fresh format-9 Flow initialization; do not resume any checkpoint from the halted custom split.
- Flow target: 25 epochs × 4,642 training pairs = 116,050 updates, subject to a 20-hour wall cap. Save every 500 updates.
- PPO starts from this new Flow checkpoint; 6-hour cap; seed 1234; 1,000,000 groups is a ceiling; save every 500 groups.
- There is no official validation set. Run the fixed training schedule without model selection on test data. Evaluate all 1,000 official test pairs once after training choices/checkpoints are fixed.
- New cache: `data/cache_format9_officialsplit_20261127`.
- New run directory: `runs/formal_mist_her2_officialsplit_20261127_seed1234`.

## Commands

```bash
python -m morphoflow_ihc prepare --manifest datasets/MIST_HER2/manifest.csv --config configs/mist_her2.json --output data/cache_format9_officialsplit_20261127
python -m morphoflow_ihc train --cache data/cache_format9_officialsplit_20261127 --config configs/mist_her2.json --output runs/formal_mist_her2_officialsplit_20261127_seed1234/flow.pt --device cuda --hours 20 --batch 1 --workers 0 --seed 1234 --save-every 500
python -m morphoflow_ihc train-rl --cache data/cache_format9_officialsplit_20261127 --checkpoint runs/formal_mist_her2_officialsplit_20261127_seed1234/flow.pt --output runs/formal_mist_her2_officialsplit_20261127_seed1234/ppo.pt --device cuda --hours 6 --groups 1000000 --batch 1 --workers 0 --seed 1234 --save-every 500
```

Do not launch the rerun until the new cache index verifies the expected counts and shows zero official-test pairs in the training entries.

Weights, images, and prediction arrays stay off Git. Training logs, split audit, checksums, environment, metrics, and final official-test summary are to be committed.
