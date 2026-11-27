# MorphoFlow-IHC — paired RGB flow prototype

Experimental paired H&E → RGB IHC generation. This branch trains conditional flow matching on corresponding H&E/IHC crops and uses the paired IHC DAB map to score regional expression during trajectory optimization. **No pathology-level accuracy, convergence, or training-time guarantee has been established.**

## Training data

MIST HER2 images and their provenance are documented in [datasets/MIST_HER2](datasets/MIST_HER2/README.md); images use Git LFS. The dataset's original license and citation requirements apply separately from this repository's code license. Training requires paired H&E/IHC rows sharing a `pair_id`. A group must not cross splits; use patient IDs where available. A filename prefix is not necessarily a patient ID.

Example training manifest (`group` examples are illustrative):

```csv
path,marker,group,domain,split,pair_id,mpp
he/p001.png,,slide-01,HE,train,pair-001,0.4661
ihc/p001.png,HER2,slide-01,IHC,train,pair-001,0.4661
he/p002.png,,slide-02,HE,val,pair-002,0.4661
ihc/p002.png,HER2,slide-02,IHC,val,pair-002,0.4661
```

`prepare` also accepts the bundled MIST provenance manifest directly. It pairs `trainA/trainB` and `valA/valB` by source prefix and filename, then deterministically repartitions whole prefixes into 80/10/10 train/validation/test groups because the archive's original train/validation image lists share prefixes. This avoids prefix overlap, but source prefixes are not verified patient IDs; do not claim patient-independent results without checking patient metadata. Only training pairs are written to the training cache. The example pixel spacing and DAB evaluation threshold should be verified for each dataset.

## Paired training

- **Conditional flow matching** interpolates Gaussian noise and a real paired IHC crop while conditioning on its corresponding H&E crop and marker.
- **Three-term generator objective** combines paired conditional flow matching, CUT-style PatchNCE, and a pix2pix-style paired PatchGAN objective. PatchNCE uses the same sampled spatial positions as positives and other positions from the same image as negatives; a small projection head adapts the fused morphology features. The PatchGAN sees H&E, IHC, and the marker indicator.
- **Segmentation-free morphology** projects H&E and IHC RGB into canonical hematoxylin structure before matching. A residual CNN with top-down FPN fusion extracts local features and three heuristic H/edge/context weights. They are not learned cell compartments or segmentation outputs. DAB is not passed to this structure encoder.
- **Policy-guided joint training** uses PPO or a direct objective with a coarse regional DAB support/amount reward computed from the paired IHC. The MIST default pools 128 px ROIs (`reward_region`); the 32 px `region` remains the fine-scale diagnostic and policy-grid size. The reward updates the policy, then the current policy mean guides the rollout used by PatchNCE and adversarial generator losses; the policy parameters are held fixed during that generator step, and the reward itself is not added as a fourth generator term. Start `--objective ppo` and `--objective direct` from the same flow checkpoint for a matched comparison.
- PPO normalizes advantages within each pair of candidate trajectories and skips pairs whose reward difference is below `min_reward_spread` (default `1e-5`). Logs expose the reward spread and informative-pair fraction so near-tie updates are not mistaken for useful learning signal.

The implementation is compared with pinned official PyTorch references in [Implementation comparison](docs/IMPLEMENTATION_COMPARISON.md) and [Upstream code and licenses](docs/UPSTREAM.md). TorchCFM is called directly; the local paired PatchNCE and PatchGAN are adaptations of the official CUT and pix2pix designs. The PPO rollout/update code is a task-specific adaptation, not an imported official trainer.

See the [logic audit](docs/LOGIC_AUDIT_20261126.md) for the verified train/update flow and the remaining pair-registration and supervision limitations.

Before a full run, follow the [pretraining preflight checklist](docs/PREFLIGHT_BEFORE_FORMAL_TRAINING_20261126.md): environment and data checks, loss/gradient tests, reward and policy checks, CUDA smoke training, and inference validation.

The paired IHC image supplies a DAB proxy map during training; no manual cell masks or expression labels are required. The trajectory reward compares DAB support fraction and amount in sliding `reward_region` windows at 50% overlap (128 px by default). Coarse pooling reduces sensitivity to local serial-section mismatch, and overlap softens grid-boundary effects. It cannot correct large or nonrigid misregistration or identify the exact positive cell. The reward is regional expression supervision; H&E-conditioned PatchNCE separately encourages structure preservation.

## Setup and run

```bash
git clone --recurse-submodules https://github.com/zhengboyu156-create/MorphoFlow-IHC.git
cd MorphoFlow-IHC
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy Pillow scipy POT pytest
python -m pip install --no-deps -e third_party/torchcfm -e third_party/torchdiffeq -e .
python -m morphoflow_ihc prepare --manifest datasets/MIST_HER2/manifest.csv --config configs/mist_her2.json --output data/cache_rgb
python -m morphoflow_ihc train --cache data/cache_rgb --config configs/mist_her2.json --output runs/rgb_flow.pt --device cuda
python -m morphoflow_ihc train-rl --cache data/cache_rgb --checkpoint runs/rgb_flow.pt --output runs/rgb_rl.pt --device cuda
python -m morphoflow_ihc infer --checkpoint runs/rgb_flow.pt --image data/he.png --marker HER2 --output prediction.png --maps prediction.npz
python -m pytest -q
```

The command writes paired RGB training caches (format 4); older unpaired caches are rejected. The current generator uses three terms: CFM, PatchNCE, and adversarial loss. Checkpoint format 8 records the paired PatchNCE projection and paired PatchGAN architecture; older checkpoints are rejected. Start joint policy training from a base flow checkpoint to initialize the policy and optimizer state.
