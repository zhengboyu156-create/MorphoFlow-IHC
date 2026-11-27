# Live training status

Updated: 2026-11-27 19:51 +08:00 (remote host clock)

- Branch: `fix/regionally-tolerant-flow-training-20261127`
- Source code commit: `cebc69021da5b911dc7185127db9fb24239230a2`
- Remote run directory: `/root/autodl-tmp/morphoflow_cebc690/runs/formal_mist_her2_format9_20261127_seed1234`
- Stage: fresh Flow training; PID 1351 active; GPU utilization 67%, 4,670 MiB used.
- Progress: 6,000 / 113,325 updates (5.3%); `flow.pt` checkpoint updated at 19:48, 49,690,081 bytes.
- Runtime at observation: about 41 minutes from launch at 19:10:09 +08:00. Overall observed throughput is about 2.44 updates/s. If it holds, Flow ETA is approximately 2026-11-28 08:02 +08:00. PPO then has a 6-hour cap, giving a provisional end near 14:02 +08:00, before full validation. Estimates may shift with throughput.
- PPO has not started in this formal run. The short preflight is not evidence of policy benefit.

Latest logged losses:

| Step | CFM | PatchNCE | Critic | Adversarial |
|---:|---:|---:|---:|---:|
| 1 | 0.21657 | 4.61299 | 0.73142 | 0.67232 |
| 500 | 0.07127 | 0.11964 | 0.000694 | 7.62258 |
| 1,000 | 0.10565 | 1.47807 | 0.001017 | 7.76663 |
| 1,500 | 0.03024 | 0.79771 | 0.000104 | 10.38404 |
| 2,000 | 0.07570 | 0.04151 | 0.65158 | 0.74667 |
| 2,500 | 0.00351 | 0.01115 | 0.74466 | 0.77915 |
| 3,000 | 0.01233 | 0.01420 | 0.86280 | 0.74492 |
| 3,500 | 0.00304 | 0.00320 | 0.69174 | 0.69976 |
| 4,000 | 0.00580 | 0.00915 | 0.69317 | 0.72887 |
| 4,500 | 0.05791 | 0.05209 | 0.62683 | 0.73322 |
| 5,000 | 0.00852 | 0.00876 | 0.70083 | 0.69996 |
| 5,500 | 0.00783 | 0.27375 | 0.65893 | 0.74001 |
| 6,000 | 0.01106 | 0.00382 | 0.67076 | 0.66664 |

All displayed values are finite, and recent critic/adversarial losses are in a relatively stable range. This indicates the logged optimization steps are executing; it does not show that any component improves held-out staining quality. Full validation, policy reward-signal analysis, and component ablations remain pending. Runtime is PyTorch 2.1.2+cu118, below the repository-declared 2.7 minimum.

No weights, images, or prediction arrays are included in this Git record.
