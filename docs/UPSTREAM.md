# Public sources

| Source | Pinned revision | Role |
| --- | --- | --- |
| [TorchCFM](https://github.com/atong01/conditional-flow-matching/tree/7c653857de4c25b979a740ef8b6a5d5ddb09b6c1) | `7c653857de4c25b979a740ef8b6a5d5ddb09b6c1` | Runtime upstream for `ConditionalFlowMatcher` and the guided-diffusion U‑Net. MIT. |
| [torchdiffeq](https://github.com/rtqichen/torchdiffeq/tree/657943acefa826ef04c025ebeb1ff5e9d60dc268) | `657943acefa826ef04c025ebeb1ff5e9d60dc268` | Runtime upstream for fixed-grid Euler integration. MIT. |
| [Flow‑GRPO](https://github.com/yifan123/flow_grpo/tree/879042cf5707f8b90daa98d147d7deac2317c5da) | `879042cf5707f8b90daa98d147d7deac2317c5da` | Research and implementation reference for windowed exploration, old log probability and clipped policy updates. MIT. |
| [BranchGRPO](https://github.com/Fredreic1849/BranchGRPO/tree/19fbb9fc92ad1857212f3b664d6e9ff80c8b3804) | `19fbb9fc92ad1857212f3b664d6e9ff80c8b3804` | Research reference for sharing generation prefixes. Apache-2.0. |

Submodules retain their upstream licenses. The two RL references are available under `third_party/`. The implementation uses a Gaussian over region-grid actions, forms a bounded DAB velocity residual, and applies the clipped policy-ratio objective to joint action log probabilities.

## Related work and dataset

- Lipman et al. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), ICLR 2023.
- Tong et al. [Improving and Generalizing Flow-Based Generative Models with Minibatch Optimal Transport](https://arxiv.org/abs/2302.00482), TMLR 2024.
- Liu et al. [Flow‑GRPO](https://arxiv.org/abs/2505.05470), NeurIPS 2025.
- He et al. [TempFlow‑GRPO](https://github.com/Shredded-Pork/TempFlow-GRPO), ICLR 2026.
- Ben Omrane et al. [CFM‑HES](https://papers.miccai.org/miccai-2026/0607-Paper5285.html), MICCAI 2026.
- Zhang et al. [PD‑UniST](https://papers.miccai.org/miccai-2025/0685-Paper1617.html), MICCAI 2025.
- Li et al. [MIST official data and code](https://github.com/lifangda01/AdaptiveSupervisedPatchNCE), MICCAI 2023.
