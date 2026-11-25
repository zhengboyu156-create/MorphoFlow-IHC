# MorphoFlow-IHC — unpaired RGB flow prototype

Experimental H&E → RGB IHC generation. This branch removes target-IHC self-conditioning and the fixed hematoxylin/DAB compositor. It adds an H&E-conditioned stain-neutral morphology rollout and a marker-conditioned IHC discriminator. Its trajectory policy scores unpaired image realism and structure proxies. **No pathology-level accuracy, convergence, or training-time guarantee has been established.**

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

The archived `datasets/MIST_HER2/manifest.csv` is a provenance manifest with a different schema; it is not directly a training manifest. Check group identity, choose leakage-safe splits, and convert it before running `prepare`. The example pixel spacing and DAB evaluation threshold should be verified for each dataset.

## Label-free training

- **Flow matching** learns IHC appearance using RGB images sampled from the IHC pool. The separate source image condition is zero for this term.
- **H&E-conditioned generation** uses the same integration grid during training and inference. A stain-neutral morphology encoder compares the input H&E with its generated image. The IHC discriminator compares the generated image with independently sampled real IHC.
- **Segmentation-free morphology** projects H&E and IHC RGB into canonical hematoxylin structure before matching. A residual CNN with top-down FPN fusion extracts high-resolution local features and three soft compartment weights. The reproducible MIST configuration uses the public, ungated DINOv2-S/14 weights as a frozen dense-token teacher. DAB is never passed to this structure encoder. Hibou-B remains an optional pathology-specific teacher for users who have accepted its gated model terms.
- **Joint trajectory training** alternates policy and flow updates. The policy samples corrections using the current flow and an appearance/structure reward. The flow then updates on its original objectives plus the reward of a new rollout using the policy mean. Start `--objective ppo` and `--objective direct` from the same flow checkpoint for a matched rollout comparison. The default `flow_reward_weight` is 0.1.

The training data consist of RGB images and target stain identity such as HER2. Stain identity selects the IHC domain. Expression grades, annotations and expression masks are absent from training. The DAB map is an optional diagnostic computed from generated RGB. Keep paired references in held-out groups for evaluation. These learning signals establish neither expression status nor cell-level localization, and the new training mode requires real-image validation.

## Setup and run

```bash
git clone --recurse-submodules https://github.com/zhengboyu156-create/MorphoFlow-IHC.git
cd MorphoFlow-IHC
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy Pillow scipy POT pytest 'transformers>=4.44,<5' safetensors
python -m pip install --no-deps -e third_party/torchcfm -e third_party/torchdiffeq -e .
python -m morphoflow_ihc prepare --manifest data/images.csv --config configs/mist.json --output data/cache_rgb
python -m morphoflow_ihc train --cache data/cache_rgb --config configs/mist.json --output runs/rgb_flow.pt --device cuda
python -m morphoflow_ihc train-rl --cache data/cache_rgb --checkpoint runs/rgb_flow.pt --output runs/rgb_rl.pt --device cuda
python -m morphoflow_ihc infer --checkpoint runs/rgb_flow.pt --image data/he.png --marker HER2 --output prediction.png --maps prediction.npz
python -m pytest -q
```

The default command writes RGB-only training caches (format 3). An old cache (format 2) can still be used if it contains both RGB pools. Training reads only its RGB arrays. Morphology-FPN flow and policy weights require format 5 checkpoints; format 4 checkpoints remain available in Git history but are intentionally rejected by this architecture. The frozen teacher is loaded only during training and is not stored in checkpoints. Start joint training from a base flow checkpoint to initialize all three optimizers. Joint checkpoints save the policy, flow and discriminator optimizer states for resuming.

`morphology_teacher: "none"` and `teacher_weight: 0` provide a lightweight ablation without downloading a foundation model. This changes the training objective and must not be mixed with the formal teacher configuration in one result row. To use Hibou-B instead, set `morphology_teacher` to `histai/hibou-b` and `morphology_teacher_dim` to `768` after obtaining Hugging Face access.
