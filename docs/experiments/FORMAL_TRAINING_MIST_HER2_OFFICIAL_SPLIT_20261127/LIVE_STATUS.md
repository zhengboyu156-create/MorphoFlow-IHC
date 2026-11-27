# Live training status

Updated: 2026-11-27 20:57:46 +08:00 (remote host clock)

- Status: official-split Flow training active; Python PID 803; started 20:53:34 +08:00.
- Source commit: `52e5c41e82d235a4892851e695d0be6f28ce022e`; branch `fix/regionally-tolerant-flow-training-20261127`.
- Audited cache: 4,642 official train / 0 validation / 1,000 official test; 0 official-test entries in training. Cache index SHA-256 `b14d501b3c60fd4593e2e3c322965be50a4d9a4537aa5979bdf1c1576f1cb1e9`.
- Progress: 500 / 116,050 Flow updates. Latest checkpoint `flow.pt` updated 20:57:05 +08:00, 49,690,081 bytes.
- Step-500 losses: CFM 0.0114914, PatchNCE 0.1350613, critic 0.0006766, adversarial 7.7660287; all finite.
- First-interval throughput: 500 updates in 4m12s (~1.98 updates/s). If sustained, Flow is estimated to finish around 2026-11-28 13:08 +08:00; PPO then has a 6-hour cap, for a provisional training end around 19:08 +08:00. The 20-hour Flow cap would not bind at this rate. This estimate is early and will be revised after more checkpoints.
- GPU utilization at observation: 62%, 4,670 MiB used. Environment: Python 3.10.8 / PyTorch 2.1.2+cu118, below the repository-declared PyTorch 2.7 minimum.
- No official validation set exists. Do not inspect the official test set for model selection; evaluate it after the schedule/checkpoint is fixed.

Weights, source images, and prediction arrays remain off Git.
