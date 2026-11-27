# Live status: MAPS-compatible official-split run

Last verified: 2026-11-27 23:39 +0800

- Source branch: `fix/maps-compatible-512-train-1024-eval-20261127`; corrected MAPS-compatible schedule commit: `51d7652be48941b12b47904c18a141c3122cd3a1`. The corrected source files are installed on the remote host.
- Dataset: official MIST HER2 `trainA/trainB`, 4,642 paired training images; `valA/valB`, 1,000 official held-out test images, excluded from training. Cache index SHA-256: `b14d501b3c60fd4593e2e3c322965be50a4d9a4537aa5979bdf1c1576f1cb1e9`.
- Runtime: Python 3.10.8, PyTorch 2.1.2+cu118, NVIDIA RTX 2080 Ti, 11,264 MiB. The PyTorch version is below the repo's declared 2.7 minimum and is an environment deviation.
- Geometry: 512x512 random paired crops, batch 1; native 1024x1024 whole-image inference.

## Flow progress and restart

- The first Flow process used a 20-hour limit and was stopped at saved step 3,500 after the user requested no time cap. Its original `flow.pt` and log are preserved; an identical checkpoint copy is `flow_timecapped_partial_step3500.pt`, SHA-256 `f957a8ffa08158195ff3120593846ea91b205ff759f455d5e60b771eea6644ef`.
- The obsolete PID 4052 run was stopped after its step 5,000 save. Its output and log remain preserved as `flow_full25_unlimited.pt` and `flow_unlimited.log`. The handoff checkpoint is `flow_pre40_resume_step5000.pt`, SHA-256 `c3d0887fd0a7e8ba094a48b64dc40e60ac3df0d2f440bce7126517c6b12fd8c1`; it contains Flow model, optimizer, and critic optimizer states.
- Corrected Flow is now running as PID 5000 with `--epochs 40 --no-time-limit --resume flow_pre40_resume_step5000.pt`, targeting 185,680 total Flow steps. At 23:39 it was computing on the RTX 2080 Ti (91% utilization, 7,088 MiB allocated). Its first periodic log after resume is expected at step 5,500.
- The corrected follow-on PPO target is 10 full passes, 46,420 groups, with policy, Flow and critic learning rates linearly decayed during that stage. This is the documented adaptation of MAPS-GAN's 40 constant + 10 decay schedule to MorphoFlow's Flow/PPO stages; it is not compute-identical to MAPS-GAN.
- Recent Flow throughput is approximately 1.2 s/step. From step 5,000, the remaining 180,680 steps project to about 60.2 hours (around 2026-11-30 noon +0800) if the rate holds. This estimates Flow only and is not a wall-clock cap. PPO time will be estimated from measured group throughput after it starts.

## Tests and held-out evaluation

- Full remote suite after the 40+10 schedule change: **21 passed**, 1 PyTorch TypedStorage deprecation warning, 16.12 s. The resumed process uses the corrected schedule code.
- Geometry smoke results are in `GEOMETRY_PREFLIGHT.md`.
- Official test-set evaluation has not started. No performance claim is made. Patient independence remains excluded from the run gate by project decision and must be recorded as a limitation.

Remote run directory:
`/root/autodl-tmp/morphoflow_maps512_20261127/runs/formal_mist_her2_maps512_officialsplit_20261127_seed1234/`

Model weights and full stdout logs remain on the remote data disk; only text protocol, test results, provenance and status are uploaded to Git.
