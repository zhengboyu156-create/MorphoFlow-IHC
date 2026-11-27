# Live training status

Updated: 2026-11-27 19:27 +08:00 (remote host clock)

- Branch: `fix/regionally-tolerant-flow-training-20261127`
- Source code commit: `cebc69021da5b911dc7185127db9fb24239230a2`
- Remote run directory: `/root/autodl-tmp/morphoflow_cebc690/runs/formal_mist_her2_format9_20261127_seed1234`
- Stage: fresh Flow training; PID 1351 active (`Rl`, 95.8% CPU); GPU utilization 56%, 4,670 MiB used.
- Progress: 2,500 / 113,325 updates; `flow.pt` checkpoint updated 19:27:04 +08:00, 49,690,081 bytes.
- Runtime at observation: 16m55s from launch at 19:10:09 +08:00. Overall observed throughput is about 2.46 updates/s. If it holds, Flow ETA is approximately 2026-11-28 08:00 +08:00. PPO then has a 6-hour cap, giving a provisional end near 14:00 +08:00, before full validation. Estimates may shift with throughput.

Latest logged losses:

| Step | CFM | PatchNCE | Critic | Adversarial |
|---:|---:|---:|---:|---:|
| 1 | 0.21657 | 4.61299 | 0.73142 | 0.67232 |
| 500 | 0.07127 | 0.11964 | 0.000694 | 7.62258 |
| 1,000 | 0.10565 | 1.47807 | 0.001017 | 7.76663 |
| 1,500 | 0.03024 | 0.79771 | 0.000104 | 10.38404 |
| 2,000 | 0.07570 | 0.04151 | 0.65158 | 0.74667 |
| 2,500 | 0.00351 | 0.01115 | 0.74466 | 0.77915 |

All displayed values are finite. Critic and adversarial losses changed sharply from steps 1,000–1,500 to 2,000–2,500; this is a monitoring signal, not by itself proof of collapse or useful generation. Full validation and PPO reward-signal analysis remain pending. Runtime is PyTorch 2.1.2+cu118, below the repository-declared 2.7 minimum.

No weights, images, or prediction arrays are included in this Git record.
