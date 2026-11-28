# MorphoFlow training and evaluation design audit

Base reviewed: `10bf1dccc145a0950115e8e22e10ca78a9935a8b`.

## Fixes

- **Training provenance.** Format 9 covered checkpoints from both the target-containing shortcut and the revised Euler loss path; format 10 added later training metadata. Current training requires format 11, which records the explicit flow objective (`regional` by default or `pixel`). Historical format 9/10 inference remains available only through the evaluator's historical-weight option. The official cache builder checks source bytes and SHA-256 against the manifest. Resuming validates the cache index, effective precision, batch size and seed. The joint stage also validates its LR horizon. Pair order, crop coordinates and update randomness resume from the recorded position. A short explicit PPO group limit can end with a partial batch. Extending that run preserves pair order and starts a new batch at the next pair.
- **Gradients and compute.** TorchCFM's custom gradient recomputation did not restore autocast. PyTorch's non-reentrant checkpoint restores the forward autocast context. The generator step freezes PatchGAN running statistics. PatchNCE's detached H&E branch no longer retains a gradient graph. PPO computes its three heuristic morphology channels without running the unused FPN branch and reuses the DAB reference calculation across candidates. The PatchNCE, adversarial and regional CFM terms keep their configured weights.
- **Reward and policy.** Regional DAB support applies the same soft threshold to output and reference, so an exact match scores 1. Four shifted grids retain within-grid non-overlap. Partial windows use only valid pixels in the region means and contribute by their valid area. PPO likelihoods use FP32 during collection and optimization. The candidate-zero win rate now measures candidate zero, sharing credit for ties. The direct objective uses the differentiable reward mean and reports AMP overflow instead of using PPO's ranking gate.
- **Evaluation.** Both scripts validate the same official test cohort, native geometry and per-file SHA-256. They accept base Flow and joint checkpoints. Bare and format 9 weights require explicit historical mode, and the report marks their objective as unverified. Prediction reuse checks the policy setting, implementation hashes, runtime, seed, inputs, and all prediction hashes. FID staging removes any extra image with an extension recognized by the backend. Fiji runs in a private ASCII work directory and writes a completion file after measuring the last image. It must also pass the ID, geometry, bit depth, area and intensity checks.

## Local verification

- RTX 5070 Ti, PyTorch 2.7.1+cu128. The installed TorchCFM and torchdiffeq checkouts matched the repository's pinned commits.
- Fresh initialization, one real paired MIST training image, centered 512 px crop, batch 1, seed 1234. A BF16 update and a subsequent FP16 update produced finite losses. FP16 did not overflow. PyTorch peak allocated memory was 1,471 MiB and 1,494 MiB respectively for these two updates. These brief measurements do not establish sustained throughput.
- On the same crop after those two updates, the PPO candidate spread was about `4.89e-6` and the update was skipped at the configured `1e-5` threshold. Recomputed action log probabilities matched the BF16 rollout's stored probabilities exactly. The 1024 px FP32 inference path with policy returned finite RGB of shape `[1, 3, 1024, 1024]`.
- Direct Fiji measured one official 1024 px IHC image (`100M2004069_10_9`, IntDen `238017601`). The launcher returned 1 and both the completion file and measurement row validated.
- All 28 CPU regression tests passed. Coverage includes bitwise equality between a two-update run and the same run interrupted and resumed after its first update, cache and checkpoint guards, and tests of the RL policy ablation metadata and reward. The official manifest contains 4,642 training and 1,000 evaluation pairs.

## Comparison protocol

Start the revised training objective from scratch with the official training cache. The published 40+10 protocol uses 40 Flow passes and 10 joint passes. For a completed experiment, run the comparison evaluator on the base Flow, the joint model with policy, and the same joint model with `--disable-policy`. Use the same seed and a unique output directory for each. The last comparison isolates the inference-time action from changes to the generator during joint training.

```bash
python artifacts/flow40_official_test_20261128/evaluate_comparison_metrics.py --weights runs/rgb_rl.pt --manifest datasets/MIST_HER2/manifest.csv --data-root datasets/MIST_HER2 --config configs/mist_her2.json --output outputs/mist_joint --device cuda --seed 1234
```

The comparison evaluator needs OpenCV, scikit-image, torchvision, pytorch-fid and Fiji with the H-DAB Colour Deconvolution plugin. Use `--fiji` to set the launcher and `--fiji-work-root` to select an ASCII path with no commas. Score definitions remain PSNR, SSIM, FID, Rc, Rp and Rave. The regional diagnostic is available from `evaluate_flow40.py` with the same checkpoint/config/manifest arguments.

## Remaining experimental questions

Regional CFM supervises pooled velocity and leaves fine spatial detail underdetermined by that term. PatchNCE supplies same-location structure supervision against the H&E input, while PatchGAN supplies IHC appearance supervision. The published test cohort can measure final comparison performance after the training choices are fixed. Whether the policy improves the final deterministic output, and whether the regional loss preserves small structures under section mismatch, require trained weights and a held-out comparison. Rc and Rp are whole-image proxies. Serial-section correspondence would require separate evidence for cell-level claims.
