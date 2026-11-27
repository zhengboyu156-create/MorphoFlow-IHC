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
- A fresh 40-epoch Flow run was launched remotely with the new algorithm at `runs/fast_endpoint_4step_mist_her2_officialsplit_20261128_seed1234/`, using FP32 on RTX 2080 Ti. At initial verification it had reached optimizer step 1 with nonzero losses and GPU utilization, so the run was active; no reliable speed or completion estimate was yet available.

## Interpretation

The unit tests establish implementation behavior, not image quality or faster end-to-end convergence. Compare measured step throughput and validation metrics against the archived 8-step run before claiming a speed or quality improvement. The run retains the official split and training schedule, but changing the endpoint estimate and solver steps creates a new algorithmic variant.
