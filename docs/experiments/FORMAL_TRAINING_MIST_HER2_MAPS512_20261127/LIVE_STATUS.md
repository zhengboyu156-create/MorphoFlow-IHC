# Live status: MAPS-compatible official-split run

Last verified: 2026-11-27 22:06:12 +0800

- Source branch: `fix/maps-compatible-512-train-1024-eval-20261127`; source commit: `a00b6968d421fc1cbb9941ac15491be9d9391984`.
- Run directory: `/root/autodl-tmp/morphoflow_maps512_20261127/runs/formal_mist_her2_maps512_officialsplit_20261127_seed1234/`.
- Stage: Flow, fresh initialization, 512x512 random paired crop, batch 1, seed 1234, official 4,642-pair train cohort; held-out 1,000-pair official test cohort is not read during training.
- GPU: RTX 2080 Ti, 11,264 MiB; at verification 7,058 MiB used and 92% utilization.
- Verified log/checkpoint: step 500; flow loss 0.0145007, PatchNCE 0.254975, critic 0.0008986, adversarial 0.0752244. Checkpoint `flow.pt` was 49,690,081 bytes and updated at 22:05:19 +0800.
- First 500 steps took 11m34s (1.388 s/step). Extrapolating to the configured 25 epochs (116,050 steps) gives about 44.7 hours. The Flow command has a 20-hour cap, which projects about 51,900 steps / 11.2 epochs if throughput stays similar; it may stop on the time cap before the 25-epoch target. This is a live estimate, not a completion claim.
- After Flow, the planned PPO stage has the configured 6-hour cap. Approximate total from launch is 26 hours plus final held-out evaluation; expected finish near 2026-11-29 00:00 +0800 if no interruptions. Re-estimate from later throughput and stage transition.
- No official test evaluation result is available yet. Patient independence is excluded from the run gate by project decision and must remain a stated limitation.

Full training stdout and checkpoints remain on the remote data disk. This Git record is a concise text status snapshot, not a substitute for the complete log or final evaluation report.