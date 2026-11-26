# Segmentation-free morphology encoder preflight

> Superseded training-data design: after review, the implementation changed to paired H&E/IHC crops, conditional flow matching, and paired regional DAB rewards. The execution observations below remain historical; the earlier unpaired limitation explains why this redesign was needed.
>
> The DINO teacher prototype described below was also retired. It is not part of the current three-term generator objective or current model implementation; these checks must not be cited as validation of the paired PatchNCE implementation.

Date: 2026-11-26 (Asia/Shanghai)

## Locked implementation

- Canonical hematoxylin projection is applied before cross-domain morphology encoding.
- The online student uses residual 3x3 BasicBlocks and top-down FPN fusion, following the public torchvision design cited in `docs/UPSTREAM.md`.
- The student consumes H concentration, Sobel gradient magnitude and absolute Laplacian.
- Nuclear-like, boundary-like and contextual weights are continuous softmax weights, not segmentation labels or masks.
- The region policy observes these three weights together with H&E, current RGB state and base velocity.
- The default frozen teacher is public `facebook/dinov2-small`. Hibou-B is not the default because its current weight download is gated.
- Teacher parameters are absent from format 5 checkpoints.

## Checks completed

- `python -m pytest -q`: 13 passed on CPU.
- Pure synthetic DAB changes while holding hematoxylin fixed changed recovered H by less than `1e-3` under the repository RGB/OD convention.
- The FPN returns a dense feature map and three finite soft weights that sum to one at each location.
- A stub frozen teacher sends finite gradients into the trainable student and teacher projection.
- Morphology reward gradients still reach the flow velocity network while scorer and policy parameters remain fixed in that update.
- The public DINOv2-S/14 checkpoint loaded without authentication and returned finite `(1, 384, 16, 16)` dense tokens for a 224 px input.
- A complete synthetic flow plus morphology forward/backward pass using the real DINOv2-S/14 teacher produced finite flow and morphology losses, a `(1, 3, 32, 32)` RGB endpoint and finite student gradients.

## Not established

This preflight checks execution and information routing. It does not show that the three soft weights equal biological nuclei, membranes or compartments. Paired supervision now provides regional expression targets, but real-image localization and generation quality still require held-out, registration-aware experiments before long training.
