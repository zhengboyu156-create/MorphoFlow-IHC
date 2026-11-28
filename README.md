# MorphoFlow-IHC — paired RGB flow prototype

Experimental paired H&E → RGB IHC generation. The method uses paired images for regional expression supervision, without treating serial-section pixels as perfectly registered. Conditional flow matching compares 64 px region averages over four grid phases, the PatchGAN sees IHC and marker only, and PPO evaluates 128 px regional DAB support/amount over four phases. **This is regional supervision, not cell-level ground truth.**

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

`prepare` also accepts the bundled MIST provenance manifest directly. The MIST authors specify 4,642 HER2 training pairs and 1,000 test pairs, with no separate official validation set; in this archive the official test cohort is named `valA/valB`. The loader preserves the official cohorts as 4,642 train / 1,000 test pairs and does not create a validation split. `source_prefix` is not treated as a patient ID. Only training pairs are written to the training cache. The official test set should be evaluated only after training choices are fixed. The example pixel spacing and DAB evaluation threshold should be verified for each dataset.

## Paired training

The MAPS-aligned image geometry is **512×512 shared-coordinate random crops for training, batch size 1, and native 1024×1024 full-image inference for evaluation**. Inference runs once over the complete canonical image; it does not tile, overlap-blend, center-crop, or resize the benchmark image when its MPP already matches the configured pixel size. This aligns input geometry with MAPS-GAN while retaining MorphoFlow's separate flow/PPO schedule. See [the protocol note](docs/MAPS_PROTOCOL_ADAPTATION_20261127.md) for exact scope and the already-running 256-crop experiment provenance.

- **Regionally tolerant conditional flow matching** interpolates Gaussian noise and paired IHC while conditioning on paired H&E and marker. Instead of pixelwise velocity MSE, it compares predicted and target velocity averages in 64 px regions over four half-window grid phases (`flow_region`).
- **Inference-aligned auxiliary training** now uses the configured noise-to-image Euler rollout from independent Gaussian noise for PatchNCE and PatchGAN, matching the generation path used at inference. This removes the earlier target-containing, one-step CFM-bridge shortcut. The corrected objective costs more computation and requires a fresh run; earlier Flow40 checkpoints and metrics remain results of the shortcut variant.
- The Flow40 code audit and its effect on historical numbers are documented in [`FLOW40_TRAIN_EVAL_AUDIT_FIXES_20261128.md`](docs/experiments/FLOW40_TRAIN_EVAL_AUDIT_FIXES_20261128.md).
- The generator objective remains regional CFM + `0.1 ×` CUT-style PatchNCE + `0.1 ×` IHC-only marker-conditioned PatchGAN. PatchNCE uses same-position H&E/generated features as positives and other positions as negatives. The discriminator receives IHC RGB and marker only, avoiding a local H&E/IHC registration cue.
- **Segmentation-free morphology** projects H&E and IHC RGB into canonical hematoxylin structure before matching. A residual CNN with top-down FPN fusion extracts local features and three heuristic H/edge/context weights. They are not learned cell compartments or segmentation outputs. DAB is not passed to this structure encoder.
- **Policy-guided joint training** uses PPO or a direct objective with a coarse regional DAB support/amount reward computed from the paired IHC. The MIST default pools 128 px ROIs (`reward_region`); the 32 px `region` remains the fine-scale diagnostic and policy-grid size. PPO updates the policy on 4-step policy-guided trajectories; the policy mean guides PatchNCE and adversarial losses during the PPO Flow update, and the reward is not added as a fourth generator term.
- PPO samples four candidate trajectories per state by default and normalizes advantages within that candidate group. A group is skipped when its max–min reward spread is below `min_reward_spread` (default `1e-5`). The count is configurable as `trajectory_candidates`; more candidates increase the chance of observing a meaningful ranking without weakening the low-signal guard. Logs expose reward spread and informative-group fraction.

The implementation is compared with pinned official PyTorch references in [Implementation comparison](docs/IMPLEMENTATION_COMPARISON.md) and [Upstream code and licenses](docs/UPSTREAM.md). TorchCFM is called directly; the local paired PatchNCE and PatchGAN are adaptations of the official CUT and pix2pix designs. The PPO rollout/update code is a task-specific adaptation, not an imported official trainer.

See the [logic audit](docs/LOGIC_AUDIT_20261126.md) for the verified train/update flow and the remaining pair-registration and supervision limitations.

Before a full run, follow the [pretraining preflight checklist](docs/PREFLIGHT_BEFORE_FORMAL_TRAINING_20261126.md): environment and data checks, loss/gradient tests, reward and policy checks, CUDA smoke training, and inference validation.

The paired IHC image supplies a DAB proxy map during training; no manual cell masks or expression labels are required. CFM compares regional velocity averages at 64 px, while PPO compares DAB support fraction and amount at the coarse `reward_region` scale (128 px by default). Both use four shifted grid phases. This reduces sensitivity to small local shifts and grid-origin effects; it cannot correct large/nonrigid mismatch or identify the exact positive cell. H&E-conditioned PatchNCE separately encourages structure preservation.

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
python -m morphoflow_ihc train --cache data/cache_rgb --config configs/mist_her2.json --output runs/rgb_flow.pt --device cuda --epochs 40 --no-time-limit
python -m morphoflow_ihc train-rl --cache data/cache_rgb --checkpoint runs/rgb_flow.pt --output runs/rgb_rl.pt --device cuda --epochs 10 --no-time-limit
python -m morphoflow_ihc infer --checkpoint runs/rgb_flow.pt --image data/he.png --marker HER2 --output prediction.png --maps prediction.npz
python -m pytest -q
```

The cache stores full canonical RGB images. For the official MIST run, rebuild earlier caches with `prepare` to verify the training image bytes against the manifest. Start Flow from scratch with a fresh output path. Format 10 records the corrected training objective and binds resumes to the cache index, batch size, seed, and effective precision. The joint checkpoint also records its LR decay horizon. Format 9 remains available for historical inference.

The [design audit](docs/experiments/FLOW_DESIGN_AUDIT_20261128.md) records the verified changes and evaluation commands for the base Flow, joint checkpoint, and same-generator policy ablation.
