# MorphoFlow-IHC — RGB + PatchNCE experimental branch

This branch replaces fixed hematoxylin/DAB compositing with a generator that outputs a complete RGB IHC image. It is an **experimental design**, not a validated method or a guarantee of cell-level spatial or biomarker accuracy. The previous DAB-compositing implementation remains in the repository history at commit `75942f8`.

## Training design

1. `prepare` reads independent H&E and IHC **training** pools from a CSV manifest (`path,marker,group,domain,split,mpp`). Group identities must not cross train/val/test. A patient ID is preferable; a filename prefix is not necessarily a patient ID. IHC marker labels are required.
2. A region predictor is supervised by DAB support derived from training IHC only. A conditional flow learns RGB IHC reconstruction with the same IHC image as its conditioning input. This self-conditioning objective does **not** itself establish how H&E predicts IHC positivity.
3. Independently sampled H&E crops condition a short, differentiable RGB flow rollout. PatchNCE compares same-position features from the input H&E and that generated output; other positions in the same image form negatives. The trainable shared encoder and projections are simple experimental components, not pretrained pathology features. No H&E–IHC random pair is treated as a matched target. No true IHC is used to compute PatchNCE.
4. The existing experimental trajectory policy is adapted to RGB. Its terminal DAB reward is derived from real training IHC and a rollout conditioned on that same IHC. Thus it is a self-reconstruction reward; its value and transfer to H&E inference remain unproven. It should not be described as direct pathology feedback on H&E, and no benefit or convergence guarantee has been established.

At inference the input H&E conditions the flow, and overlapping RGB tiles are blended. DAB is optionally extracted **after** generation solely for diagnostic maps; it is not composited into the output. No real target IHC is used for inference.

This model's IHC-conditioned flow and H&E-conditioned PatchNCE have a significant domain/identifiability gap: realistic RGB IHC distribution, correct positive location, structure preservation and policy benefit have not been demonstrated. Tune only on a disjoint validation split and evaluate on independent held-out patients. If patient identity cannot be established, do not claim patient-independent generalization.

## Data and installation

MIST HER2 images are available via [datasets/MIST_HER2](datasets/MIST_HER2/README.md) and Git LFS. Dataset licensing and citations follow the original provider, not the code license.

```bash
git clone --recurse-submodules https://github.com/zhengboyu156-create/MorphoFlow-IHC.git
cd MorphoFlow-IHC
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy Pillow scipy POT pytest
python -m pip install --no-deps -e third_party/torchcfm -e third_party/torchdiffeq -e .
```

## Usage

Install PyTorch and pinned torchcfm/torchdiffeq submodules as in the earlier revision, then install this project. Rebuild caches; RGB models use checkpoint/cache format 2 and do not load old format 1 weights.

```bash
python -m morphoflow_ihc prepare --manifest data/images.csv --config configs/mist.json --output data/cache_rgb
python -m morphoflow_ihc train --cache data/cache_rgb --config configs/mist.json --output runs/rgb_flow.pt --device cuda
python -m morphoflow_ihc train-rl --cache data/cache_rgb --checkpoint runs/rgb_flow.pt --output runs/rgb_rl.pt --device cuda
python -m morphoflow_ihc infer --checkpoint runs/rgb_flow.pt --image data/he.png --marker HER2 --output prediction.png --maps prediction.npz
python -m pytest -q
```

Both training domains must be supplied in the manifest. DAB thresholds and MPP in example configurations are engineering defaults and are not validated pathology or clinical thresholds.
