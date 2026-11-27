# Live training status

Updated: 2026-11-27 20:54 +08:00 (remote host clock)

- Status: official-split Flow training active; PID 803; started 20:53:34 +08:00.
- Source commit: `52e5c41e82d235a4892851e695d0be6f28ce022e`; branch `fix/regionally-tolerant-flow-training-20261127`.
- Data split is audited: 4,642 official train pairs, 0 validation pairs, 1,000 official test pairs; zero official-test pairs in the train cache. See `CACHE_AUDIT.md`.
- Progress: step 1 / 116,050. Fresh `flow.pt` created at 20:53:45 +08:00, 49,690,081 bytes. Step-1 losses are finite: CFM 0.29265, PatchNCE 4.47229, critic 0.73835, adversarial 0.66964.
- Run settings: seed 1234, batch 1, workers 0, 25 epochs, 20-hour Flow cap; PPO follows this fresh checkpoint with a 6-hour cap. No checkpoint from the halted custom split was resumed.
- At launch the GPU showed 49% utilization and 4,670 MiB used. Runtime is Python 3.10.8 / PyTorch 2.1.2+cu118, below the repository's declared PyTorch 2.7 minimum.
- Provisional estimate using the prior run's measured 2.57 updates/s: Flow ~12h33, PPO up to 6h, total training ~18h33 from launch. Refresh after the first 500 steps; official test evaluation is additional and not timed yet.
- No official validation set exists. Test data will be evaluated only after training choices are fixed.

Weights, image files, and prediction arrays are not included in Git.
