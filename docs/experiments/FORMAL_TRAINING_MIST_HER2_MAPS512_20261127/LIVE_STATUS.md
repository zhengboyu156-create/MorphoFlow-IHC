# Live status: MAPS-compatible official-split run

Last verified: 2026-11-27 22:17:13 +0800

- Source branch: `fix/maps-compatible-512-train-1024-eval-20261127`; source commit: `a00b6968d421fc1cbb9941ac15491be9d9391984`.
- Run directory: `/root/autodl-tmp/morphoflow_maps512_20261127/runs/formal_mist_her2_maps512_officialsplit_20261127_seed1234/`.
- Stage: Flow, fresh initialization, 512x512 random paired crop, batch 1, seed 1234, official 4,642-pair train cohort; held-out 1,000-pair official test cohort is not read during training.
- GPU: RTX 2080 Ti, 11,264 MiB; at latest verification 7,058 MiB used and 93% utilization.
- Verified checkpoints/logs: step 500 (22:05:19 +0800) and step 1,000 (22:16:04 +0800); checkpoint `flow.pt` is 49,690,081 bytes. At step 1,000, logged losses were flow 0.124673, PatchNCE 0.560105, critic 0.0020249, adversarial 0.0826664.
- Step 500 -> 1,000 took about 11m01s (1.322 s/step). Losses are stochastic per batch and rose at step 1,000 relative to step 500; two sampled log points do not establish convergence.
- The configured target is 25 epochs (116,050 steps), but the Flow command has a 20-hour cap. At current throughput this projects about 54,500 steps / 11.7 epochs by the time cap, so it may stop before 25 epochs. Estimated Flow time-cap time is 2026-11-28 about 17:54 +0800. These are estimates, not guarantees.
- After Flow, the planned PPO stage has the configured 6-hour cap. Approximate total from launch is 26 hours plus final held-out evaluation; expected finish near 2026-11-29 00:00 +0800 if there are no interruptions. Re-estimate from later throughput and stage transition.
- No official test evaluation result is available yet. Patient independence is excluded from this run's gate by project decision and must remain a stated limitation.

Full training stdout and checkpoints remain on the remote data disk. This Git record is a concise text status snapshot, not a substitute for the complete log or final evaluation report.