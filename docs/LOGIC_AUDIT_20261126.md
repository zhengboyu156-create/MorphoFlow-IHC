# MorphoFlow-IHC logic audit — 2026-11-26

## Verdict

The main training path is now connected end to end: paired H&E/IHC data train a conditional velocity field; the regional policy is optimized from a paired-IHC DAB reward; and the updated policy then guides the rollout used by the generator's PatchNCE and adversarial terms. The generator still has exactly three terms (CFM, PatchNCE, adversarial). This is an implementation-level loop, not evidence that the method improves staining or identifies biological expression correctly.

## Verified data path

- The bundled MIST manifest has 11,284 image rows: 5,642 H&E/IHC filename pairs across 61 source prefixes. All referenced files exist and every pair key has one H&E and one IHC image.
- The corrected MIST path preserves all 4,642 author-designated training pairs and all 1,000 author-designated test pairs. The source archive names its official test folders `valA/valB`; source prefixes are not treated as patient identifiers. MIST has no separate official validation cohort, so the test set remains reserved for final evaluation.
- Training samples a shared crop coordinate from each paired image and checks matching canonical dimensions. That establishes filename-level pairing and shape consistency, not pixel registration.

## Verified objective and update path

1. **CFM:** TorchCFM samples an interpolation between noise and paired IHC. The U-Net predicts velocity conditioned on the corresponding H&E crop and marker.
2. **PatchNCE:** fixed stain-neutral morphology channels feed a trainable residual/FPN encoder. Shared sampled positions are positives; other sampled positions from the same image are negatives. The feature map is a structure proxy, not a segmentation or cell-type label.
3. **PatchGAN:** the discriminator receives only real/generated IHC RGB and marker identity. Its own real/fake objective updates the discriminator; the generator's adversarial term differentiates through the discriminator into the generated image.
4. **Policy:** PPO samples four corrections by default from a shared prefix/noise rollout and scores them with regional paired-reference DAB support/amount. The bounded score updates only the policy in that update. Older checkpoints without a stored candidate-count setting retain the legacy two-candidate behavior.
5. **Generator feedback:** after the policy update, the next generator step uses the policy mean in the same rollout implementation as inference. Policy parameters are excluded from that optimizer step, but gradients through the corrected trajectory reach the velocity generator. CFM continues to train the base vector field; PatchNCE and adversarial terms train the policy-guided output. This closes the prior code gap where the policy was updated but omitted from generator rollouts.

## Remaining scientific assumptions

- The data loader uses the same crop coordinates for filename-paired H&E and IHC, but serial sections can be locally misregistered. The revised CFM averages velocity targets over 64 px regions and four grid phases; the IHC-only critic no longer uses H&E/IHC pair structure. The 128 px PPO reward remains paired and regional. These choices reduce sensitivity to small local shifts but do not correct large/nonrigid displacement. Describe the method as paired-region supervision; do not claim exact-cell correspondence.
- The paired DAB reward comes from the same IHC target used for CFM. It is a regional objective, not an independent biological label. Its reward-policy signal should be evaluated on held-out pairs before claiming policy benefit.
- The soft morphology weights are heuristic functions of H concentration and edge strength. They are not learned or validated nuclear/membrane compartments.
- No held-out performance, convergence, robustness or clinical HER2 conclusion follows from these implementation checks.

## Validation status

Python bytecode compilation and `git diff --check` pass. Runtime tests could not be executed in this environment because PyTorch, TorchCFM, torchdiffeq and pytest are unavailable. Run `python -m pytest -q` in the project environment before training.
