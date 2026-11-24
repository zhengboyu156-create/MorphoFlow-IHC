# MorphoFlow-IHC — unpaired RGB flow prototype

Experimental H&E → RGB IHC generation. This branch removes target-IHC self-conditioning and the fixed hematoxylin/DAB compositor. It adds an H&E-conditioned PatchNCE rollout and a marker-conditioned IHC discriminator. Its trajectory policy scores unpaired image realism and structure proxies. **No pathology-level accuracy, convergence, or training-time guarantee has been established.**

## Training data

MIST HER2 images and their provenance are documented in [datasets/MIST_HER2](datasets/MIST_HER2/README.md); images use Git LFS. The dataset's original license and citation requirements apply separately from this repository's code license. Training requires separate H&E and IHC image pools, plus marker information for IHC. A CSV manifest has fields `path,marker,group,domain,split,mpp`; H&E training marker may be blank. Paths may be relative to the CSV. A group must not cross splits; use patient IDs where available. A filename prefix is not necessarily a patient ID.

Example training manifest (HE and IHC are independent pools; `group` examples are illustrative):

```csv
path,marker,group,domain,split,mpp
he/p001.png,,p001,HE,train,0.4661
ihc/p002.png,HER2,p002,IHC,train,0.4661
he/p003.png,,p003,HE,val,0.4661
ihc/p003.png,HER2,p003,IHC,val,0.4661
```

The archived `datasets/MIST_HER2/manifest.csv` is a provenance manifest with a different schema; it is not directly a training manifest. Check group identity, choose leakage-safe splits, and convert it before running `prepare`. The example MPP and DAB threshold are unverified for a given dataset; establish them from metadata/training data before drawing conclusions.

## What the prototype trains

- **Region predictor:** training IHC H and DAB proxies from the same image supervise an IHC-domain regional support predictor. Its transfer to H&E and its biomarker meaning remain unverified.
- **Flow:** samples Gaussian noise and a real IHC target and regresses the standard unconditional noise-to-IHC velocity. At this loss the image condition and regional probability are zero. The interpolated flow state contains the target as in ordinary flow matching, but the separate condition does not. This is not matched H&E supervision.
- **H&E conditioning:** generate an RGB IHC candidate through a short H&E-conditioned flow rollout; PatchNCE compares same-location H&E/output features. A marker-conditioned discriminator compares independent real IHC with this output. The discriminator loss trains its own network; generator objectives are flow, regional support, PatchNCE and adversarial distribution matching. No random H&E/IHC pair is treated as a matched target.
- **Policy:** freeze the flow, patch encoder, and discriminator. At one rollout step the policy modifies RGB velocity with separate bounded per-channel actions; two candidates receive a bounded combination of the frozen discriminator score and shared-index patch-feature similarity. This is an **image realism/structure proxy**, not a reward for correct positive cells, and can be exploited by the policy. It does not use unmatched IHC as pixel truth. Its benefit, stability and generalization remain to be tested.

Nonpaired marginal distributions, PatchNCE and adversarial realism alone do not identify which cells express a biomarker. This branch does not establish reliable HER2/KI67 localization or any clinical use. Weak labels or other valid expression supervision and an independent evaluation would be needed to substantiate that claim. Actual structure preservation is also not guaranteed.

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
python -m morphoflow_ihc prepare --manifest data/images.csv --config configs/mist.json --output data/cache_rgb
python -m morphoflow_ihc train --cache data/cache_rgb --config configs/mist.json --output runs/rgb_flow.pt --device cuda
python -m morphoflow_ihc train-rl --cache data/cache_rgb --checkpoint runs/rgb_flow.pt --output runs/rgb_rl.pt --device cuda
python -m morphoflow_ihc infer --checkpoint runs/rgb_flow.pt --image data/he.png --marker HER2 --output prediction.png --maps prediction.npz
python -m pytest -q
```

The manifest must include both train HE and train IHC images. Cache format 2 is retained; new checkpoints use **format 3** and reject prior checkpoint formats because the objective and model changed. The image output is directly generated RGB; the optional DAB map is calculated from it for diagnostics only. `nce_steps` is a short training rollout and can differ from inference `ode_steps`.

Original compositor code remains in Git history; the prior RGB/PatchNCE self-conditioning prototype remains at `3a4a4e283cdafcc6763f4c9e8b7d7a11eabf8bf4` (PR #1).

## Weak expression supervision for unpaired images

Use independent H&E and IHC case groups. Add these fields to the training manifest:

| Column | Values |
| --- | --- |
| `label` | Binary expression endpoint, `0` or `1` |
| `label_source` | `independent_assay` for H&E, `ihc_annotation` for real IHC. Record the actual provenance externally. |
| `expression_scope` | `region` for a whole preselected region or `slide` for a whole slide image |
| `expression_endpoint` | The same explicit binary endpoint identifier in both domains for each marker |

Specify the marker on every labeled row, including H&E rows. Each marker needs both classes in each domain's train and validation pools. Reuse the true patient/group ID across all rows and keep groups disjoint across splits and between training domains. The loader checks declared identity and provenance. Confirm actual case identity, label provenance and endpoint comparability before running. Labels copied from the corresponding MIST images would change the supervision protocol. The MIST archive has no independent H&E expression labels in its published manifest.

```bash
python -m morphoflow_ihc prepare --manifest expression.csv --config configs/mist_her2.json --output data/expression_cache
python -m morphoflow_ihc train-expression --manifest expression.csv --config configs/mist_her2.json --output runs/scorers.pt
python -m morphoflow_ihc train --cache data/expression_cache --config configs/mist_her2.json --expression-manifest expression.csv --expression runs/scorers.pt --output runs/expression_flow.pt
python -m morphoflow_ihc train-rl --cache data/expression_cache --expression-manifest expression.csv --checkpoint runs/expression_flow.pt --objective ppo --output runs/policy_ppo.pt
python -m morphoflow_ihc train-rl --cache data/expression_cache --expression-manifest expression.csv --checkpoint runs/expression_flow.pt --objective direct --output runs/policy_direct.pt
python -m morphoflow_ihc infer --checkpoint runs/expression_flow.pt --image region.png --marker HER2 --expression-scope region --output prediction.png --maps prediction.npz
```

The scorer artifact includes per-domain, per-marker validation metrics. Training checks that both scorers beat a constant training-prevalence predictor in validation Brier score and rank the positive-class mean above the negative-class mean. This is a basic validation gate. Measure performance separately on independent patients and account for repeated units per patient. Both scorers are frozen during generation training. Scope applies to the whole image supplied to the scorer and to the generated unit. Inputs are resized and padded as whole units, including slide scope, which limits recoverable slide detail. A region score describes the whole region and is exported as a scalar. The optional DAB map describes the generated pixels separately.

Start PPO and direct optimization from the same flow checkpoint and use equal rollout budgets. Test the base flow, PPO and direct optimization with a separate labeled test set, including any registered IHC localization data appropriate to the intended claims.
