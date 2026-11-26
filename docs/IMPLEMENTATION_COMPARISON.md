# Official-code comparison and task-specific adaptation

All three generator objectives are implemented with **PyTorch tensors, modules, autograd and optimizers**. Their code patterns can therefore share one training step, but their source APIs are not interchangeable. TorchCFM is called directly; PatchNCE and PatchGAN are local adaptations modeled on pinned official references.

| Objective | Official reference pattern | Current implementation | Task-specific change |
|---|---|---|---|
| Conditional flow matching | TorchCFM samples a conditional interpolation and target vector field. | `StainFlow.losses` calls `ConditionalFlowMatcher(sigma=0.0)` and regresses the conditional velocity. | The velocity U-Net receives the paired H&E crop and marker as conditions; the target endpoint is the paired IHC crop. |
| PatchNCE | CUT samples feature-map locations, projects local feature vectors, uses corresponding patches as positives and other patches as negatives. | `PatchSampleF` and `PatchNCELoss` in `morphoflow_ihc/patchnce.py`. One shared index set is used for H&E keys and generated-IHC queries; negatives are within each image. | The features come from stain-neutral morphology FPN maps. Positives align the generated image to its H&E input; this term does not depend on reference-IHC registration. |
| Adversarial | pix2pix concatenates the condition and target for a multi-scale local PatchGAN decision and trains with a GAN objective. | `IHCCritic` in `morphoflow_ihc/model.py` follows the N-layer PatchGAN convolution pattern; `flow_update` uses logistic softplus losses. | The discriminator input is H&E + IHC + a marker indicator map. This checks local paired appearance for the requested stain. |

Pinned implementation files for line-by-line review:

- TorchCFM matcher: [`torchcfm/conditional_flow_matching.py`](https://github.com/atong01/conditional-flow-matching/blob/7c653857de4c25b979a740ef8b6a5d5ddb09b6c1/torchcfm/conditional_flow_matching.py); the paired conditional velocity network is called from `morphoflow_ihc/model.py`.
- CUT PatchNCE: [`models/patchnce.py`](https://github.com/taesungp/contrastive-unpaired-translation/blob/b3ac297708dfb6f7589d04662277e53c0d579c27/models/patchnce.py), [`PatchSampleF` in `models/networks.py`](https://github.com/taesungp/contrastive-unpaired-translation/blob/b3ac297708dfb6f7589d04662277e53c0d579c27/models/networks.py), and assembly in [`models/cut_model.py`](https://github.com/taesungp/contrastive-unpaired-translation/blob/b3ac297708dfb6f7589d04662277e53c0d579c27/models/cut_model.py).
- pix2pix conditional discriminator and GAN loss: [`NLayerDiscriminator` and `GANLoss` in `models/networks.py`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2a7afba2895d52556dd5dfe07e8555ef657ced6f/models/networks.py), assembled with paired input/target in [`models/pix2pix_model.py`](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2a7afba2895d52556dd5dfe07e8555ef657ced6f/models/pix2pix_model.py).

All three run as PyTorch modules and losses under the same local optimizer step, but they are not one official shared framework: TorchCFM is a direct library call; PatchNCE and pix2pix GAN pieces are reimplemented locally to fit RGB IHC, marker conditioning, paired crops, and the morphology feature encoder. PPO is a fourth, separate policy objective and references Flow-GRPO; it is not one of the three generator-loss terms.

## Objective layout

The generator objective has three terms:

\[
\mathcal L_G=\mathcal L_{\mathrm{CFM}}+\lambda_{\mathrm{NCE}}\mathcal L_{\mathrm{PatchNCE}}+\lambda_{\mathrm{adv}}\mathcal L_{\mathrm{adv}}.
\]

The discriminator has its own real/fake PatchGAN objective. PPO has a separate policy objective and paired regional-DAB reward; that reward is not added again to the generator loss. After the policy update, its mean action guides the rollout used for the generator's PatchNCE and adversarial terms, allowing those gradients to update the flow network through the corrected trajectory. The policy's parameters are held fixed for that generator step. Thus the policy participates in alternating joint training without becoming a fourth generator-loss term.

## What is and is not copied

- TorchCFM's matcher and U-Net are upstream package code.
- CUT and pix2pix source repositories are pinned Git submodules for inspection and provenance. The local PatchNCE/PatchGAN code is a task-specific adaptation, not a direct import of their complete models.
- The local PatchNCE keeps CUT's projection/sampling/positive-negative structure but ties positive pairs to the same spatial coordinates because the dataset is paired.
- The local discriminator keeps pix2pix's conditional PatchGAN pattern while adding marker conditioning and a compact channel width.
- No CUT generator, pix2pix generator, dataset loader, or full training launcher is substituted for the flow model.

Official references: [TorchCFM](https://github.com/atong01/conditional-flow-matching/tree/7c653857de4c25b979a740ef8b6a5d5ddb09b6c1), [CUT](https://github.com/taesungp/contrastive-unpaired-translation/tree/b3ac297708dfb6f7589d04662277e53c0d579c27), and [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/2a7afba2895d52556dd5dfe07e8555ef657ced6f).
