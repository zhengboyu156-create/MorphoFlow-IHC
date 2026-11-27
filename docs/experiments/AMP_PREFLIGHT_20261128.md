# Mixed-precision speed preflight — 2026-11-28

## Purpose

The RTX 2080 Ti run was taking about 1.17 seconds per 512 px Flow update. This preflight checks whether FP16 autocast can reduce runtime without changing the loss terms, image geometry, or optimizer schedule.

## Hardware and software

- GPU: NVIDIA GeForce RTX 2080 Ti, 11 GiB, compute capability 7.5.
- Runtime: PyTorch 2.1.2+cu118.
- BF16 is unsupported on this GPU; before this change the training path therefore used FP32.

## Results

- The code now has an explicit `--amp` mode, GradScaler state is checkpointed, and CPU/non-AMP training remains supported. Full remote test suite: **22 passed**, 1 upstream TypedStorage deprecation warning.
- A 100-step FP16 smoke with activation checkpointing enabled failed during backward: TorchCFM's custom checkpoint recomputation did not restore autocast, causing a Half input / Float bias mismatch.
- Disabling U-Net activation checkpointing avoided that dtype mismatch but exceeded available memory: allocation reached 10.31 GiB, with only 5 MiB free when another 32 MiB was requested. No FP16 smoke checkpoint was produced.
- The formal FP32 checkpoint at step 10,000 remains intact (`flow_prespeed_step10000_fp32.pt`, SHA-256 `76c56aef76bb725b76b2b49f380583ce6229d0fb9df809725b79b10423614d2d`). Formal training resumed from it in FP32 with the original 40+10 target and unchanged loss coefficients.

## Conclusion

FP16 is not viable for this 512 px setup on the current 11 GiB GPU with TorchCFM's activation checkpoint implementation. Auto precision falls back to BF16 only when supported and otherwise FP32. Explicit FP16 remains an experimental option and should not be used for this run.
