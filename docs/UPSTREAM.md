# Public implementation references

| Public source | Pinned revision | Role in MorphoFlow-IHC | How this code adapts it |
|---|---|---|---|
| [TorchCFM](https://github.com/atong01/conditional-flow-matching/tree/7c653857de4c25b979a740ef8b6a5d5ddb09b6c1) | `7c653857de4c25b979a740ef8b6a5d5ddb09b6c1` | Runtime source for `ConditionalFlowMatcher` and guided-diffusion U-Net. MIT. | Use the official matcher and U-Net; add H&E and marker conditioning for paired staining. |
| [CUT](https://github.com/taesungp/contrastive-unpaired-translation/tree/b3ac297708dfb6f7589d04662277e53c0d579c27) | `b3ac297708dfb6f7589d04662277e53c0d579c27` | Official PyTorch PatchNCE, patch sampling/projection and contrastive translation reference. BSD-3-Clause. | Adapt same-position positives and same-image negatives to paired H&E/IHC morphology-FPN features. |
| [pix2pix / CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/2a7afba2895d52556dd5dfe07e8555ef657ced6f) | `2a7afba2895d52556dd5dfe07e8555ef657ced6f` | Official PyTorch paired cGAN, PatchGAN discriminator and GAN objective reference. BSD-3-Clause for pix2pix files. | Concatenate H&E with IHC and marker indicator maps; use the paired PatchGAN pattern and logistic objective. |
| [torchdiffeq](https://github.com/rtqichen/torchdiffeq/tree/657943acefa826ef04c025ebeb1ff5e9d60dc268) | `657943acefa826ef04c025ebeb1ff5e9d60dc268` | Runtime upstream for fixed-grid Euler integration. MIT. | Integrate the conditional velocity field on a deterministic grid. |
| [Flow-GRPO](https://github.com/yifan123/flow_grpo/tree/879042cf5707f8b90daa98d147d7deac2317c5da) | `879042cf5707f8b90daa98d147d7deac2317c5da` | Research/implementation reference for rollout exploration, old log probabilities and clipped policy updates. MIT. | Adapt region-grid actions and paired regional DAB reward; not imported as a ready-made trainer. |
| [BranchGRPO](https://github.com/Fredreic1849/BranchGRPO/tree/19fbb9fc92ad1857212f3b664d6e9ff80c8b3804) | `19fbb9fc92ad1857212f3b664d6e9ff80c8b3804` | Research reference for sharing generation prefixes. Apache-2.0. | The current one-step policy rollout shares its prefix before candidate corrections. |
| [torchvision ResNet/FPN](https://github.com/pytorch/vision/tree/9eb57cd5c96be7fe31923eb65399c3819d064587/torchvision) | `9eb57cd5c96be7fe31923eb65399c3819d064587` | Architectural reference for local residual blocks and top-down feature pyramid. BSD-3-Clause. | Local implementation keeps only the residual/FPN pattern; no torchvision source is vendored. |

CUT and pix2pix are included as pinned Git submodules under `third_party/` for direct source comparison. Initialize them with `git submodule update --init --recursive`. TorchCFM, torchdiffeq, Flow-GRPO and BranchGRPO submodules retain their upstream licenses. The code in `morphoflow_ihc/patchnce.py` and the local `IHCCritic` are adaptations, not copies of the upstream modules; see `docs/IMPLEMENTATION_COMPARISON.md`.

## Related research and dataset

- Lipman et al. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), ICLR 2023.
- Tong et al. [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482), TMLR 2024.
- Liu et al. [Flow-GRPO](https://arxiv.org/abs/2505.05470), NeurIPS 2025.
- He et al. [TempFlow-GRPO](https://github.com/Shredded-Pork/TempFlow-GRPO), ICLR 2026.
- Ben Omrane et al. [CFM-HES](https://papers.miccai.org/miccai-2026/0607-Paper5285.html), MICCAI 2026.
- Zhang et al. [PD-UniST](https://papers.miccai.org/miccai-2025/0685-Paper1617.html), MICCAI 2025.
- Li et al. [MIST official data and code](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE), MICCAI 2023.
