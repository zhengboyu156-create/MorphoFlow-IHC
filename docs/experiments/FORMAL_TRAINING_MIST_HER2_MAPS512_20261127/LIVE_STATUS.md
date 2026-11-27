# Live status: MAPS-compatible official-split run

Last verified: 2026-11-27 22:47:25 +0800

- Source branch: `fix/maps-compatible-512-train-1024-eval-20261127`; source commit: `a00b6968d421fc1cbb9941ac15491be9d9391984`.
- Run directory: `/root/autodl-tmp/morphoflow_maps512_20261127/runs/formal_mist_her2_maps512_officialsplit_20261127_seed1234/`.
- Stage: Flow, fresh initialization, 512x512 random paired crop, batch 1, seed 1234, official 4,642-pair train cohort; held-out 1,000-pair official test cohort is not read during training.
- GPU: RTX 2080 Ti, 11,264 MiB; at latest verification 7,058 MiB used and 92% utilization.
- Verified checkpoints/logs: steps 500 (22:05:19), 1,000 (22:16:04), 1,500 (22:26:46), 2,000 (22:37:15), and 2,500 (22:47:00), all +0800. Checkpoint `flow.pt` is 49,690,081 bytes.
- Step 2,000 -> 2,500 took about 9m45s (1.17 s/step). Latest logged losses at step 2,500: flow 0.040495, PatchNCE 0.304256, critic 0.0001704, adversarial 0.099683. Losses vary by random batch; these points do not establish convergence.
- The configured target is 25 epochs (116,050 steps), but the Flow command has a 20-hour cap. Using the latest interval projects about 61,500 steps / 13.2 epochs by the time cap, so it may stop before 25 epochs. A simple throughput projection for the full 25 epochs is about 37.6 hours. Estimated Flow time-cap time is 2026-11-28 about 17:54 +0800. These are estimates, not guarantees.
- After Flow, the planned PPO stage has the configured 6-hour cap. Approximate total from launch is 26 hours plus final held-out evaluation; expected finish near 2026-11-29 00:00 +0800 if there are no interruptions. Re-estimate from later throughput and stage transition.
- No official test evaluation result is available yet. Patient independence is excluded from this run's gate by project decision and must remain a stated limitation.

Full training stdout and checkpoints remain on the remote data disk. This Git record is a concise text status snapshot, not a substitute for the complete log or final evaluation report.