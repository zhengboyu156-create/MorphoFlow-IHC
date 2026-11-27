# Full MIST HER2 training run (format 9)

## Provenance

- Repository branch: `fix/regionally-tolerant-flow-training-20261127`.
- Code commit: `cebc69021da5b911dc7185127db9fb24239230a2` (the current branch also contains preflight result archive commits after this code commit).
- Dataset manifest: `datasets/MIST_HER2/manifest.csv`, SHA-256 `bd667a845ddf15071815c6f31a042d190abc353ecf2c14f5cfc5f6d8c6695571`.
- Source inventory: 11,284 rows / 2,465,298,285 image bytes. The source tarball was 2,475,318,784 bytes and matched local SHA-256 `22bc5d2d135db4a5572d2a3043592c00d4a94c74cce480b913d40a00f14bbe13`; every manifest-listed image matched both recorded byte count and SHA-256 on the remote host.
- Preparation created a new format-4 cache from the complete manifest: 4,533 train / 600 validation / 509 test pairs across 61 source-prefix groups. Only training images were written into the cache. The 16-pair smoke cache was not used.
- Grouping uses `source_prefix`; patient-level independence is not claimed or required for this run.
- Runtime: AutoDL RTX 2080 Ti (11 GiB), Python 3.10.8, PyTorch 2.1.2+cu118. This differs from the repository's declared PyTorch 2.7 minimum and is recorded as an environment limitation.

## Fixed run settings

- Config: `configs/mist_her2.json`, SHA-256 `e33a580361e04555922150e2779070135099bd356ed77c6f341c4b4ea9dc1c0d`.
- Seed: 1234; batch size 1; workers 0.
- Flow: fresh format-9 initialization, no resume, 20-hour budget; checkpoint `flow.pt`, save every 500 steps.
- PPO: starts from this run's new `flow.pt`, 6-hour budget, seed 1234; `groups=1,000,000` is a ceiling so the time budget controls the stage; checkpoint `ppo.pt`, save every 500 groups.
- Remote run directory: `/root/autodl-tmp/morphoflow_cebc690/runs/formal_mist_her2_format9_20261127_seed1234`.
- Flow launch timestamp: `2026-11-27 19:10:09 +08:00`. Planned end of the two training stages is approximately `2026-11-28 21:10 +08:00`, before full validation; this estimate will be updated from observed throughput.

## Status at launch

The first Flow update was logged with finite CFM, PatchNCE, critic, and adversarial losses. A new 48 MB checkpoint was saved, and the GPU showed active compute. No formal result claim is made until both training stages and full validation complete. Weights, images, and prediction arrays will remain off Git; logs, metrics, configuration, hashes, and provenance will be archived.
