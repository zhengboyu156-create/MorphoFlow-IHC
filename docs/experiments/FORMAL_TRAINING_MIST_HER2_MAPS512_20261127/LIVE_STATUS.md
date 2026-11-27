# Live status: MAPS-compatible official-split run

Last verified: 2026-11-28 01:00 +0800

- Source branch: `fix/maps-compatible-512-train-1024-eval-20261127`; corrected MAPS-compatible schedule commit: `51d7652be48941b12b47904c18a141c3122cd3a1`. The corrected source files are installed on the remote host.
- Dataset: official MIST HER2 `trainA/trainB`, 4,642 paired training images; `valA/valB`, 1,000 official held-out test images, excluded from training. Cache index SHA-256: `b14d501b3c60fd4593e2e3c322965be50a4d9a4537aa5979bdf1c1576f1cb1e9`.
- Runtime: Python 3.10.8, PyTorch 2.1.2+cu118, NVIDIA RTX 2080 Ti, 11,264 MiB. The PyTorch version is below the repo's declared 2.7 minimum and is an environment deviation.
- Geometry: 512x512 random paired crops, batch 1; native 1024x1024 whole-image inference.

## Flow progress and restart

- The first Flow process used a 20-hour limit and was stopped at saved step 3,500 after the user requested no time cap. Its original `flow.pt` and log are preserved; an identical checkpoint copy is `flow_timecapped_partial_step3500.pt`, SHA-256 `f957a8ffa08158195ff3120593846ea91b205ff759f455d5e60b771eea6644ef`.
- The obsolete PID 4052 run was stopped after its step 5,000 save. Its output and log remain preserved as `flow_full25_unlimited.pt` and `flow_unlimited.log`. The handoff checkpoint is `flow_pre40_resume_step5000.pt`, SHA-256 `c3d0887fd0a7e8ba094a48b64dc40e60ac3df0d2f440bce7126517c6b12fd8c1`; it contains Flow model, optimizer, and critic optimizer states.
- Corrected Flow is running as PID 5000 with `--epochs 40 --no-time-limit --resume flow_pre40_resume_step5000.pt`, targeting 185,680 total Flow steps. At 01:00 it was active on the RTX 2080 Ti (92% utilization, 7,088 MiB allocated). Latest saved checkpoint is step 9,000 (2026-11-28 00:56:13 +0800); the process continues beyond that save.
- Step-9,000 batch-local losses: CFM 0.020651, PatchNCE 0.006675, critic 0.728995, adversarial 0.577980. These stochastic minibatch values are not convergence evidence.
- The corrected follow-on PPO target is 10 full passes, 46,420 groups, with policy, Flow and critic learning rates linearly decayed during that stage. This is the documented adaptation of MAPS-GAN's 40 constant + 10 decay schedule to MorphoFlow's Flow/PPO stages; it is not compute-identical to MAPS-GAN.
- Measured saved-checkpoint interval from step 5,000 (23:38:29) to step 9,000 (00:56:13) is about 1.166 s/step. From step 9,000, the remaining 176,680 Flow steps project to about 57.2 hours, around 2026-11-30 10:15 +0800 if that rate holds. This is an estimate, not a wall-clock cap.
- PPO target remains 46,420 groups. No PPO throughput is measured yet; a code-call-count estimate is roughly 23-30 hours for PPO, so the full run projection is about 80-87 hours from the 01:00 check (around 2026-12-01 09:00-16:00 +0800). Replace this provisional estimate after the first PPO save interval.

## Tests and held-out evaluation

- Full remote suite after the 40+10 schedule change: **21 passed**, 1 PyTorch TypedStorage deprecation warning, 16.12 s. The resumed process uses the corrected schedule code.
- Geometry smoke results are in `GEOMETRY_PREFLIGHT.md`.
- Official test-set evaluation has not started. No performance claim is made. Patient independence remains excluded from the run gate by project decision and must be recorded as a limitation.

Remote run directory:
`/root/autodl-tmp/morphoflow_maps512_20261127/runs/formal_mist_her2_maps512_officialsplit_20261127_seed1234/`

Model weights and full stdout logs remain on the remote data disk; only text protocol, test results, provenance and status are uploaded to Git.
