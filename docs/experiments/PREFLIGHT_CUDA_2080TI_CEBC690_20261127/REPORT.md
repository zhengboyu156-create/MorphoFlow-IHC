# 2080 Ti CUDA preflight — cebc690 (2026-11-27)

## Provenance and environment

- Code: `fix/regionally-tolerant-flow-training-20261127` at `cebc69021da5b911dc7185127db9fb24239230a2`.
- Runtime: Python 3.10.8; PyTorch 2.1.2+cu118; CUDA-enabled NVIDIA GeForce RTX 2080 Ti (11 GiB; driver 560.35.05, CUDA driver API 12.6).
- Isolated venv packages: torchcfm 1.0.7, torchdiffeq 0.2.5, pytest 8.4.2, pluggy 1.7.0, POT 0.9.7.post1; system-site packages include NumPy 1.26.4 and Pillow 10.3.0. The system PyTorch was retained. This is below the repository's declared `torch>=2.7` floor; observed checks passed on this alternate runtime, but version parity with the declared environment is not established.
- The first environment check found torchcfm, torchdiffeq, and pytest missing; pytest then exposed an old system pluggy and torchcfm required POT. The missing packages were added to a new `/root/autodl-tmp/morphoflow-preflight-env` venv; the existing system environment was not changed.
- Training cache: 16 paired training crops sourced from the already prepared local format-4 cache. One fixed validation pair was used (`VALIDATION_PAIR.txt`). No source images, predictions, arrays, or checkpoint weights are archived here.
- The source image inventory manifest has SHA-256 `bd667a845ddf15071815c6f31a042d190abc353ecf2c14f5cfc5f6d8c6695571`; the reused cache index itself does not store that hash, so direct manifest-to-cache linkage is not independently proven by this run.

## Results

- Full test suite: **20 passed, 1 PyTorch TypedStorage deprecation warning**, 75.87 s. Raw output: `pytest.txt`.
- CUDA flow smoke: **10/10 updates**; CFM, PatchNCE, critic, and adversarial losses were finite at every step. Format-9 checkpoint saved and reloaded by downstream PPO and inference. Raw log and checkpoint SHA-256 metadata are in `flow_smoke.log` and `summary.json`; the weight itself stays on the remote host.
- CUDA PPO smoke: **8/8 groups had `signal_fraction=1.0`**, each accepted 2 PPO passes, all logged values finite. Group reward spreads ranged from `1.061e-5` to `9.489e-5` with the configured `1e-5` gate; the minimum is only slightly above the cutoff. KL ranged from `1.247e-6` to `2.280e-6`. This confirms execution and distinguishable candidates for this tiny smoke cache, not robust learning signal.
- Inference loaded both flow and policy checkpoints. Both returned finite 1024×1024 RGB output for the 1024×1024 input. With the same seed, repeated policy inference was pixel-identical. Base vs policy output mean absolute RGB difference was 15.923/255 and 97.15% of pixels changed.
- On the single fixed validation pair, 32 px regional support MAE changed from 0.41740 (base) to 0.39146 (policy); 128 px reward-scale support MAE changed from 0.41587 to 0.38913. This one-pair improvement is descriptive only and cannot establish reward/error alignment or policy quality.

## Decision and limitations

The 20 tests, 10-step CUDA flow smoke, 8-group PPO smoke, format-9 checkpoint reload, output-shape check, and same-seed repeat all passed. This is a useful implementation smoke on a 2080 Ti. It is **not** a formal-training release: only 16 training pairs and one validation pair were used; the smallest PPO spread barely cleared the low-signal threshold; no multi-pair reward-direction analysis, visual seam review, or extended stability run was performed. Patient independence was not evaluated, as requested. No formal 20-hour flow training or 6-hour PPO training was started.
