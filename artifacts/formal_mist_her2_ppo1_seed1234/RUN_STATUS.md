# Formal MIST HER2 training run

- Locked source: formal/mist-her2-locked-ppo-20261129 at 082482984fa57a7de83f510b315b720ec57e2010.
- The run began locally on the RTX 5070 Ti using the locked config and a fresh cache.
- Cache verification: 4,642 official_train pairs; the index contains no test image entries. The separate 1,000-pair official_test cohort remains excluded from training.
- Flow target: 40 epochs / 185,680 updates. PPO target: 10 epochs / 46,420 groups, automatically scheduled only after a successful Flow completion.
- Effective precision: BF16 (torch.bfloat16), seed 1234, batch 1, crop 512, velocity limit 1.0, min reward spread 1e-5.
- At this snapshot Flow reached step 1,000/185,680. flow_loss=0.013969, nce_loss=2.666068, adversarial_loss=0.129055.
- An early throughput estimate projected about 19 hours remaining for Flow plus 5-6 hours for PPO; this is provisional and will be revised from later checkpoints.
- Checkpoints, source data and generated cache are kept on local D: storage and are not included in Git. The official test set has not been evaluated.
- C: initially ran out of space during an incomplete cache build. The incomplete cache has no index, was moved under D:\2222\data, and is not used.
- Full machine-local provenance is in run_manifest.json; training and preparation logs are under logs/.

## Storage relocation and resume

All source images, image caches and pilot/formal weights are physically under D:\2222. The C: workspace paths datasets, data and runs are junctions to D:\2222 so repository commands keep working. Flow was paused at a saved step-2,000 checkpoint, its optimizer state was confirmed, then resumed from that checkpoint using the same cache index and locked settings. At this update it is running again from step 2,000; no test images were included in the cache.

