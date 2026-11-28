# Formal MIST HER2 training run

- Locked source: formal/mist-her2-locked-ppo-20261129 at 082482984fa57a7de83f510b315b720ec57e2010.
- The run began locally on the RTX 5070 Ti using the locked config and a fresh cache.
- Cache verification: 4,642 official_train pairs; the index contains no test image entries. The separate 1,000-pair official_test cohort remains excluded from training.
- Flow target: 40 epochs / 185,680 updates. PPO target: 10 epochs / 46,420 groups, automatically scheduled only after a successful Flow completion.
- Effective precision: BF16 (torch.bfloat16), seed 1234, batch 1, crop 512, velocity limit 1.0, min reward spread 1e-5.
- At this snapshot Flow reached step 1,000/185,680. flow_loss=0.013969, nce_loss=2.666068, adversarial_loss=0.129055.
- An early throughput estimate projected about 19 hours remaining for Flow plus 5-6 hours for PPO; this is provisional and will be revised from later checkpoints.
- Checkpoints, source data and generated cache are kept on local D: storage and are not included in Git. The official test set has not been evaluated.
- C: ran out of space while an initial cache was being built at the workspace path. That incomplete cache has no index and is not used. The run was rebuilt on D:. Recursive cleanup was blocked by the environment's automatic review, so the incomplete C: cache remains until manually removed.
- Full machine-local provenance is in run_manifest.json; training and preparation logs are under logs/.
