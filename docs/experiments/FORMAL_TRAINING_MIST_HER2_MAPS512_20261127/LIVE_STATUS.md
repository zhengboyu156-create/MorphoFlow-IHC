# Live status: MAPS-compatible official-split run

Last verified: 2026-11-27 22:37:56 +0800

- Source branch: `fix/maps-compatible-512-train-1024-eval-20261127`; source commit: `a00b6968d421fc1cbb9941ac15491be9d9391984`.
- Run directory: `/root/autodl-tmp/morphoflow_maps512_20261127/runs/formal_mist_her2_maps512_officialsplit_20261127_seed1234/`.
- Stage: Flow, fresh initialization, 512x512 random paired crop, batch 1, seed 1234, official 4,642-pair train cohort; held-out 1,000-pair official test cohort is not read during training.
- GPU: RTX 2080 Ti, 11,264 MiB; at latest verification 7,058 MiB used and 91% utilization.
- Verified checkpoints/logs: steps 500 (22:05:19), 1,000 (22:16:04), 1,500 (22:26:46), and 2,000 (22:37:15), all +0800. Checkpoint `flow.pt` is 49,690,081 bytes.
- Step 1,500 -> 2,000 took about 10m29s (1.258 s/step). Latest logged losses at step 2,000: flow 0.052555, PatchNCE 0.475197, critic 0.0004219, adversarial 0.092286. Losses vary by random batch; these points do not establish convergence.
- The configured target is 25 epochs (116,050 steps), but the Flow command has a 20-hour cap. Using the recent interval projects about 57,000 steps / 12.3 epochs by the time cap, so it may stop before 25 epochs. Estimated Flow time-cap time is 2026-11-28 about 17:54 +0800. These are estimates, not guarantees.
- After Flow, the planned PPO stage has the configured 6-hour cap. Approximate total from launch is 26 hours plus final held-out evaluation; expected finish near 2026-11-29 00:00 +0800 if there are no interruptions. Re-estimate from later throughput and stage transition.
- No official test evaluation result is available yet. Patient independence is excluded from this run's gate by project decision and must remain a stated limitation.

Full training stdout and checkpoints remain on the remote data disk. This Git record is a concise text status snapshot, not a substitute for the complete log or final evaluation report.