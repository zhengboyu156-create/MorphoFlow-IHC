# Non-Overlapping Grid-Phase Preflight (2026-11-27)

## Implementation and provenance

- Code commit tested: `883ab1b6b42d18ce365014454b9eb72586d6d91d`
- Branch target: `fix/overlap-tolerant-regional-reward-20261127`
- Reward and evaluator score four phase-shifted grids. Each phase uses stride equal to window size, so windows do not overlap within a phase. Phase scores are averaged; different phases can cover the same tissue at shifted boundaries.
- `reward_region=128 px` is the PPO reward/evaluation scale; `region=32 px` remains a fine-scale diagnostic.
- Runtime: Python 3.12.4, PyTorch 2.7.1+cu128, CUDA 12.8, NVIDIA GeForce RTX 5070 Ti.
- Existing MIST cache and weights remained local; none are included in this archive.

## Results

- Full test suite: 19 passed (`pytest.txt`), including synthetic exact/small/distant shift ordering and the shifted non-overlapping grid test.
- RTX PPO smoke: 8 groups, batch 1, seed 1234. 4/8 groups had signal; remaining groups were skipped by the low-signal guard. Raw records are in `ppo_smoke.jsonl`.
- One fixed validation pair was inferred with base and policy checkpoints using the same H&E and seed. At 32 px diagnostic scale, error was 0.4279 (base) and 0.4165 (policy). At the matching 128 px reward scale, error was 0.4267 (base) and 0.4150 (policy). Raw metrics are in `validation_base.json` and `validation_policy.json`.
- The exploratory structural displacement audit is in `displacement_qc.json`; it did not yield reliable local displacement estimates. Fixed sample list: `validation_subset.json`.

## Decision

This confirms code tests, PPO startup, and train/evaluation metric execution. The eight-group smoke and one validation pair are too small to establish reward effectiveness. Structural correspondence remains unresolved. Formal training remains NO-GO pending broader validation and usable local correspondence QC.
