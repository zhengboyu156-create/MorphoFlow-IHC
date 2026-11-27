# Official split cache audit

- Source commit: `52e5c41e82d235a4892851e695d0be6f28ce022e`.
- Source manifest SHA-256: `bd667a845ddf15071815c6f31a042d190abc353ecf2c14f5cfc5f6d8c6695571`.
- Full manifest: 11,284 image rows, 5,642 filename pairs; cohort reconstruction found 4,642 official training pairs and 1,000 official test pairs.
- Cache preparation command exited 0 at 2026-11-27 20:51 +08:00.
- Cache path: `/root/autodl-tmp/morphoflow_officialsplit_20261127/data/cache_format9_officialsplit_20261127`.
- Cache counts: 4,642 train / 0 validation / 1,000 test.
- `source_cohort_split_counts`: official_train -> 4,642 train; official_test -> 1,000 test; all other cohort/split counts zero.
- Training cache entries: 4,642; all are `official_train` and `train`. Official test entries in training: **0**.
- Training payload files: 9,284 NPZ arrays (H&E and IHC for each training pair).
- Cache index SHA-256: `b14d501b3c60fd4593e2e3c322965be50a4d9a4537aa5979bdf1c1576f1cb1e9`.
- A remote audit script asserted the official counts, source-cohort cross-tab, exact pair-ID coverage, and zero official-test entries in training before launch.

MIST has no separate official validation set. The official test cohort remains reserved for final evaluation.
