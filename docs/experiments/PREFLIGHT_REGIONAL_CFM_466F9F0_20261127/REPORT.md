# Regional CFM preflight: `466f9f0` (2026-11-27)

## Scope and provenance

- Requested source commit: `466f9f0f665490979b81482c12a6b861c3c44f13` (`fix/regionally-tolerant-flow-training-20261127`).
- Verification was performed from that commit in a separate worktree. The code and test changes in the resulting commit fix findings from this preflight.
- No model training, CUDA smoke training, inference, or checkpoint generation was started in this pass.
- Test outputs, environment metadata, and this report are text-only. No source images, cache arrays, predictions, or weights are included.

## Findings and correction

The first full test run on the unmodified source commit produced **9 failed, 10 passed**. The main defect was that `Config.flow_region` defaulted to 64 px even when a test config used a 32 px crop; configuration validation then rejected the run before model construction. That contradicted the supported tiny-crop smoke configuration.

The default is now selected as the largest region-aligned divisor of `crop` no greater than 64 px. Thus the default remains 64 px for the production `crop=256, region=32` configuration and resolves to 32 px for `crop=32, region=8`. A regression test covers both. An existing checkpoint test also expected loading format 8 despite this branch explicitly changing the flow/critic objective and requiring fresh format 9 checkpoints; it now asserts that format 8 is rejected.

After the corrections, the full suite reports **20 passed**. `git diff --check` passed, and `python -m morphoflow_ihc --help` completed successfully.

## Local data and hardware checks

- Runtime: Python 3.12.4, PyTorch 2.7.1+cu128, pytest 7.4.4; NVIDIA GeForce RTX 5070 Ti, 16,303 MiB.
- Dataset inventory: 11,284 image rows; manifest SHA-256 `bd667a845ddf15071815c6f31a042d190abc353ecf2c14f5cfc5f6d8c6695571`.
- Existing paired cache: format 4, 4,533 train / 600 validation / 509 test pairs; config stain, marker, and pixel-spacing settings match. All 9,066 train payload files exist, and a sample loads with marker id plus two `(3, 256, 256)` tensors.
- The cache index does not contain the manifest hash, so its direct provenance link to the current manifest is not independently recorded here. The split grouping is by source prefix; patient-level independence is not claimed, per the user's instruction.
- At the time of inspection, GPU utilization was 86% and 7,384 MiB was in use by a separately identified MAPS-V3 training process. No MorphoFlow GPU smoke was launched to avoid interfering with that run.

## Decision

**Not yet a CUDA preflight pass and not a formal-training release.** The code-level test failures are fixed, and the local cache is readable. The required CUDA flow smoke, PPO smoke on this objective, checkpoint reload, and inference/tile checks remain unverified for `466f9f0` plus the correction. Recheck GPU availability before running those steps; do not count preflight results from other commits as evidence for this objective.
