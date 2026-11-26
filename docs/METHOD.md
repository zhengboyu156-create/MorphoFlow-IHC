# Paired conditional RGB flow

Each training sample is a filename-level H&E/IHC pair. Both crops use the same coordinate. The code rejects missing pair members, mismatched groups/splits, and unequal canonical image dimensions.

## Generator: three objectives

The generator is trained with

\[
\mathcal L_G=\mathcal L_{\mathrm{CFM}}+\lambda_{\mathrm{NCE}}\mathcal L_{\mathrm{PatchNCE}}+\lambda_{\mathrm{adv}}\mathcal L_{\mathrm{adv}}.
\]

- **Conditional flow matching** uses the paired IHC as the endpoint and its corresponding H&E and marker as conditions. The implementation calls TorchCFM's `ConditionalFlowMatcher` and uses its guided-diffusion U-Net.
- **PatchNCE** follows the official CUT organization: sample shared spatial indices, project local features, treat same-position H&E/generated-IHC features as positives, and other positions within the image as negatives. It compares the input H&E with its own generated output; it does not use reference-IHC pixel alignment. The exact CUT modules remain available as a pinned submodule for side-by-side comparison.
- **Adversarial loss** follows the paired pix2pix PatchGAN structure. The discriminator receives concatenated H&E, IHC and marker indicator maps. The adaptation uses a marker-conditioned paired discriminator and logistic loss.

No separate teacher-distillation or generator reward term is included. The configured MIST weights start at `nce_weight=0.1` and `adversarial_weight=0.1`; they require empirical tuning and are not claimed optimal.

## Morphology features

Both images are projected into a canonical hematoxylin concentration by cancelling the domain's other stain vector: eosin for H&E and DAB for IHC. A residual CNN/FPN processes H concentration, Sobel magnitude and Laplacian channels. Its three returned weights are fixed soft proxies computed from H concentration and edge strength; they are not biological compartments or segmentation masks. DAB is not an input to this encoder.

## PPO policy

The policy makes one bounded regional velocity correction at a selected flow step. Its separate reward compares generated and paired-reference DAB support and amount over regional windows. PPO updates the policy from this score; the score is not added a second time to the generator loss. On each subsequent generator update, the current policy's mean action is used in the rollout that supplies PatchNCE and adversarial losses. Policy parameters are held fixed during this generator step, while gradients pass through the corrected trajectory into the flow network. CFM continues to regularize the base velocity field. Therefore the full model has three generator terms, one discriminator objective, and one policy objective. The reward is still derived from the paired IHC and is not independent biological supervision.

## Data split and limitations

The bundled MIST provenance manifest can be used directly. Since original image-level train/validation lists share source prefixes, the loader deterministically assigns whole prefixes to 80/10/10 train/validation/test sets before caching. This avoids prefix overlap; source prefixes are not verified patient IDs. Filename pairing does not guarantee exact serial-section registration. Registration quality matters for pixelwise CFM targets, paired adversarial conditioning and regional DAB reward. PatchNCE anchors generated morphology features to H&E coordinates but does not establish cell identity or expression status. The model learns paired-image statistics; it does not establish causal cell-level expression understanding.
