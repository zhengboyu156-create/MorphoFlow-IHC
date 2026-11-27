# Live training status

Updated: 2026-11-27 19:24 +08:00 (remote host clock)

- Branch: `fix/regionally-tolerant-flow-training-20261127`
- Source code commit: `cebc69021da5b911dc7185127db9fb24239230a2`
- Remote run directory: `/root/autodl-tmp/morphoflow_cebc690/runs/formal_mist_her2_format9_20261127_seed1234`
- Stage: fresh Flow training; PID 1351 was active (`Sl`, 93.6% CPU); GPU utilization 63%, 4,670 MiB used.
- Progress: 1,500 / 113,325 updates; latest checkpoint `flow.pt` updated 19:24:00 +08:00, 49,690,081 bytes.
- Runtime at observation: 13m43s from launch at 19:10:09 +08:00.
- Recent throughput: steps 1,000 to 1,500 in about 176 s (2.84 updates/s). At that rate, Flow ETA is approximately 2026-11-28 06:20 +08:00. PPO then has a 6-hour cap, giving a provisional end near 12:20 +08:00, before full validation. Estimates may shift with throughput.

Latest logged losses:

| Step | CFM | PatchNCE | Critic | Adversarial |
|---:|---:|---:|---:|---:|
| 1 | 0.21657 | 4.61299 | 0.73142 | 0.67232 |
| 500 | 0.07127 | 0.11964 | 0.000694 | 7.62258 |
| 1,000 | 0.10565 | 1.47807 | 0.001017 | 7.76663 |
| 1,500 | 0.03024 | 0.79771 | 0.000104 | 10.38404 |

All displayed values are finite. The critic loss is low and adversarial loss is high at recent checkpoints; this is a monitoring signal, not by itself proof of collapse or useful generation. Full validation and PPO reward-signal analysis remain pending. Runtime is PyTorch 2.1.2+cu118, below the repository-declared 2.7 minimum.

No weights, images, or prediction arrays are included in this Git record.
