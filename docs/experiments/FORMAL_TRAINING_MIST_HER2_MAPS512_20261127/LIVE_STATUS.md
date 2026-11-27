# Live status: MAPS-compatible official-split run

Last verified: 2026-11-27 23:08:51 +0800

- Source branch: `fix/maps-compatible-512-train-1024-eval-20261127`; code commit with unlimited mode: `358c1a4d5eac86a13cc2be2b5e4bec5d28e835c4`.
- Dataset: official MIST HER2 `trainA/trainB`, 4,642 paired training images; `valA/valB`, 1,000 official held-out test images, excluded from training. Cache index SHA-256: `b14d501b3c60fd4593e2e3c322965be50a4d9a4537aa5979bdf1c1576f1cb1e9`.
- Runtime: Python 3.10.8, PyTorch 2.1.2+cu118, NVIDIA RTX 2080 Ti, 11,264 MiB. The PyTorch version is below the repo's declared 2.7 minimum and is an environment deviation.
- Geometry: 512x512 random paired crops, batch 1; native 1024x1024 whole-image inference.

## Flow progress and restart

- The first Flow process used a 20-hour limit and was stopped at saved step 3,500 after the user requested no time cap. Its original `flow.pt` and log are preserved; an identical checkpoint copy is `flow_timecapped_partial_step3500.pt`, SHA-256 `f957a8ffa08158195ff3120593846ea91b205ff759f455d5e60b771eea6644ef`.
- The new process (PID 4052 at last check) resumes from step 3,500 into `flow_full25_unlimited.pt` using `--no-time-limit`. It will stop at the configured 25-epoch target, 116,050 total optimizer steps, not at a wall-clock deadline. At last check it had started and allocated 7,088 MiB; the next periodic log/checkpoint is expected at step 4,000.
- Latest completed saved point before restart: step 3,500 at 23:06:36 +0800; flow loss 0.024348, PatchNCE 0.030846, critic 0.598912, adversarial 0.082268. These are batch-local stochastic losses, not convergence evidence.
- Recent step intervals were about 1.17-1.32 s/step. From step 3,500, the remaining 112,550 steps project around 36.6 hours if that rate holds. This is an estimate only; no wall-clock limit is set.
- After Flow, PPO will run to its configured 3,000 groups with `--no-time-limit`; no hours cap will be passed. Its remaining duration will be estimated from observed group throughput when that stage starts.

## Tests and held-out evaluation

- Full suite after adding `--no-time-limit`: **20 passed**, 1 PyTorch TypedStorage deprecation warning, 72.67 s while Flow was running concurrently.
- Geometry smoke results are in `GEOMETRY_PREFLIGHT.md`.
- Official test-set evaluation has not started. No performance claim is made. Patient independence remains excluded from the run gate by project decision and must be recorded as a limitation.

Remote run directory:
`/root/autodl-tmp/morphoflow_maps512_20261127/runs/formal_mist_her2_maps512_officialsplit_20261127_seed1234/`

Model weights and full stdout logs remain on the remote data disk; only text protocol, test results, provenance and status are uploaded to Git.