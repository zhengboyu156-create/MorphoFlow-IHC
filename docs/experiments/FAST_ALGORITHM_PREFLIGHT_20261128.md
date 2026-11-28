# Fast algorithm preflight (2026-11-28)

## Code and protocol

- Branch: `fast/endpoint-cfm-4step-ppo-20261128`
- Code commit: `34246bfdcc725bed34a1d5e0572f3b4471db6b3c`
- Official MIST HER2 training split, 4,642 pairs; 512 x 512 crops; batch 1; seed 1234.
- Schedule retained: 40 Flow epochs followed by 10 PPO epochs; CFM, PatchNCE, and GAN losses and their configured weights are retained.
- During the Flow-only stage, reuse the existing CFM velocity prediction to form a single-step endpoint estimate for PatchNCE and adversarial auxiliary losses. For a straight bridge this is exact when the velocity is exact; for the learned field it is an approximation. The CFM objective itself is unchanged.
- PPO and inference use four Euler steps instead of eight; PPO retains four candidates. This cuts trajectory evaluations, with a possible quality tradeoff that requires evaluation.
- The change is not checkpoint-equivalent to the previous eight-step algorithm. A new run is required.

## Verification

- Remote command: `/root/autodl-tmp/morphoflow-preflight-env/bin/python -m pytest -q tests/test_pipeline.py`
- Result: **23 passed**, one PyTorch `TypedStorage` deprecation warning, in 20.62 seconds.
- Local `compileall` and `git diff --check` passed.
- The previous eight-step run was stopped after saving step 13,000/185,680. Its checkpoint and log were preserved under `archive_old_8step_13000_20261128/` on the remote host; checkpoint SHA-256: `a22e948f753957febdc506413924f73a1753839e3c1a417664d472f11de5d9dd`.
- A fresh 40-epoch Flow run was launched remotely with the new algorithm at `runs/fast_endpoint_4step_mist_her2_officialsplit_20261128_seed1234/`, using FP32 on RTX 2080 Ti. At step 1,000 it had nonzero losses and active GPU computation. The first 1,000 steps took about 287 seconds including startup (about 0.287 s/step); the 500-1,000 interval was also about 3 minutes. At this rate, the remaining 184,680 Flow steps are estimated at about 14.7 hours, or around 2026-11-28 17:10 Asia/Shanghai. This is an early estimate from a short window, not a completed-run duration; PPO throughput has not yet been measured, so full 40+10 completion time remains unknown.

## Interpretation

The unit tests establish implementation behavior, not image quality or faster end-to-end convergence. The initial Flow interval is about 4.1x faster than the archived run's observed approximately 1.17 s/step, but longer-window throughput and validation metrics are needed before treating this as a stable speed or quality improvement. The run retains the official split and training schedule, but changing the endpoint estimate and solver steps creates a new algorithmic variant.

## Superseding audit note (2026-11-28)

The endpoint shortcut described above was later found to use the CFM bridge state as the generated image for PatchNCE/PatchGAN, so the auxiliary losses did not see the same noise-to-image distribution as inference. The corrected code removes this shortcut and uses the configured differentiable Euler rollout for auxiliary losses. The shortcut's speed measurements and Flow40 checkpoint remain historical; they do not validate or represent the corrected training objective. See `FLOW40_TRAIN_EVAL_AUDIT_FIXES_20261128.md` for the full audit and fresh-run requirement.
