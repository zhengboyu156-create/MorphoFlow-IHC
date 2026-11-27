# Live training status

Updated: 2026-11-27 21:01:43 +08:00 (remote host clock)

- Status: official-split Flow training active; Python PID 803; started 20:53:34 +08:00.
- Source commit: `52e5c41e82d235a4892851e695d0be6f28ce022e`; branch `fix/regionally-tolerant-flow-training-20261127`.
- Audited cache: 4,642 official train / 0 validation / 1,000 official test; zero official-test pairs in training. Cache index SHA-256: `b14d501b3c60fd4593e2e3c322965be50a4d9a4537aa5979bdf1c1576f1cb1e9`.
- Progress: 1,000 / 116,050 Flow updates. Latest checkpoint `flow.pt` updated 21:00:18 +08:00, 49,690,081 bytes.
- Step-1 losses: CFM 0.29265, PatchNCE 4.47229, critic 0.73835, adversarial 0.66964.
- Step-500 losses: CFM 0.01149, PatchNCE 0.13506, critic 0.000677, adversarial 7.76603.
- Step-1,000 losses: CFM 0.05341, PatchNCE 0.22543, critic 0.000393, adversarial 8.77780. Logged values are finite; loss values alone do not establish quality or convergence.
- Throughput: steps 500→1,000 took about 193 s (~2.59 updates/s); overall average through step 1,000 is about 2.48 updates/s. If sustained, remaining Flow is about 12h54 and should finish near 2026-11-28 09:55 +08:00. PPO then has a 6-hour cap; provisional training completion is near 15:55 +08:00. Estimates will be revised from later checkpoints.
- GPU utilization at observation: 58%, 4,670 MiB used. Runtime: Python 3.10.8 / PyTorch 2.1.2+cu118, below the repository-declared PyTorch 2.7 minimum.
- There is no official validation set. Official test data remains reserved for final evaluation after training choices are fixed.

Weights, source images, and prediction arrays remain off Git.
