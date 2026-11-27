# Regionally tolerant conditional RGB flow

Each training sample is a filename-level H&E/IHC pair. Both crops use the same canonical coordinate. The loader rejects missing pair members, mismatched groups/splits, and unequal canonical image dimensions. Pair identity does not imply pixel-perfect registration, so the flow objective is regionally pooled and the realism critic does not receive H&E.

## Generator: three objectives

The generator is trained with

\[
\mathcal L_G=\mathcal L_{\mathrm{CFM}}+\lambda_{\mathrm{NCE}}\mathcal L_{\mathrm{PatchNCE}}+\lambda_{\mathrm{adv}}\mathcal L_{\mathrm{adv}}.
\]

- **Conditional flow matching** uses paired IHC as endpoint and H&E plus marker as conditions. The implementation calls TorchCFM's `ConditionalFlowMatcher` and uses its guided-diffusion U-Net. It minimizes a four-phase regional velocity error,
  \[
  \mathcal L_{\mathrm{CFM}}^{\mathrm{reg}}=\frac{1}{4}\sum_{p\in\{(0,0),(0,r/2),(r/2,0),(r/2,r/2)\}}\left\|\operatorname{AvgPool}_{r,p}(v_\theta)-\operatorname{AvgPool}_{r,p}(u)\right\|_2^2,
  \]
  with `r=flow_region` (64 px by default). This supervises regional expression instead of exact pixel values. Phase averaging reduces grid-origin effects but does not correct large tissue displacement.
- **PatchNCE** follows the official CUT organization: sample shared spatial indices, project local features, treat same-position H&E/generated-IHC features as positives, and other positions within the image as negatives. It compares the input H&E with its own generated output; it does not use reference-IHC pixel alignment. The exact CUT modules remain available as a pinned submodule for side-by-side comparison.
- **Adversarial loss** uses a marker-conditioned IHC-only PatchGAN with logistic loss. It matches stain appearance without using local H&E/IHC registration as a realism cue.

No separate teacher-distillation or generator reward term is included. The configured MIST weights start at `nce_weight=0.1` and `adversarial_weight=0.1`; they require empirical tuning and are not claimed optimal.

## Morphology features

Both images are projected into a canonical hematoxylin concentration by cancelling the domain's other stain vector: eosin for H&E and DAB for IHC. A residual CNN/FPN processes H concentration, Sobel magnitude and Laplacian channels. Its three returned weights are fixed soft proxies computed from H concentration and edge strength; they are not biological compartments or segmentation masks. DAB is not an input to this encoder.

## PPO policy

The policy makes one bounded regional velocity correction at a selected flow step. Its separate reward compares generated and paired-reference DAB support and amount over overlapping coarse windows (`reward_region`, 128 px by default), rather than requiring cell-wise coordinate matches. PPO samples four candidate trajectories sharing the same prefix by default. Advantages are normalized within each candidate group; a group is skipped when its max–min reward range is below `min_reward_spread` (default `1e-5`) and reported through `signal_fraction`. This prevents near-ties from being treated as strong policy evidence. The score is not added a second time to the generator loss. On each subsequent generator update, the current policy's mean action is used in the rollout that supplies PatchNCE and adversarial losses. Policy parameters are held fixed during this generator step, while gradients pass through the corrected trajectory into the flow network. CFM continues to regularize the base velocity field. Therefore the full model has three generator terms, one discriminator objective, and one policy objective. The reward is still derived from the paired IHC and is not independent biological supervision.

## Data split and limitations

The MIST authors specify 4,642 HER2 training pairs and 1,000 test pairs, with no separate validation set; the bundled archive calls the official test folders `valA/valB`. The loader preserves all 4,642 training pairs and all 1,000 test pairs. Source prefixes are not used as patient identifiers. The test cohort is reserved for final evaluation after training choices are fixed. The regional CFM and PPO reward reduce sensitivity to small shifts. Large/nonrigid mismatch still weakens supervision and cannot be repaired by these losses. The method supervises regional expression, not cell identity or exact positivity location. PatchNCE encourages morphology preservation but does not establish expression status. The model learns paired regional statistics; it does not establish causal cell-level expression understanding.
