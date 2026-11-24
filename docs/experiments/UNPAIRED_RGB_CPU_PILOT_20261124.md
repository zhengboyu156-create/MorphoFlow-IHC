# Nonpaired RGB flow: CPU pilot (2026-11-24)

## Scope

Exploratory functionality and directional check, not a formal experiment. Data were drawn from the public MIST HER2 archive at commit `8214f4205b82fbe096ae4c08e3cc80ce9331df2d`. Eight HE and eight IHC train images, and three held-out HE/IHC pairs were selected. The eight train prefixes and three validation prefixes are disjoint, but patient-level independence was not established. Crop 64 px, batch 2, seed 1234, CPU PyTorch 2.8.0, marker HER2, DAB proxy threshold 0.35; 128 px validation center crops. Example physical pixel spacing is a configuration assumption, not independently verified. Image alignment and stain decomposition can affect proxy measures.

## Model and schedules

The unpaired flow loss used zero conditioning and a real IHC target. HE conditioning was applied on the separate differentiable PatchNCE plus adversarial rollout. The discriminator and unpaired region-score components were trained as in the source in this commit. Training ran 100 steps then resumed to 300; resume reset the random seed, so this should not be treated as a bitwise reproduction of a single uninterrupted 300-step run. The 300-step checkpoint was then frozen for policy tests.

The default 3-channel policy was trained for 200 groups with scale `0.0001`, learning rate `1e-5`. A separate exploratory higher learning-rate run used `0.001`, same scale, seed, base checkpoint, 200 groups and KL limit 0.02. This higher setting is **not** the default and should not be regarded as selected on independent data. Two candidates share the same noise, prefix and PatchNCE negative locations. The unpaired reward uses only a frozen style discriminator and structure proxy, not true biomarker location.

## Results

| Validation sample prefix | Real IHC support | Flow step 300 support | Flow step 300 window MAE | Policy at lr=0.001 window MAE |
|---|---:|---:|---:|---:|
| 48M2101314 | 0.0207 | 0.2509 | 0.2302 | 0.2385 |
| 79M2100417 | 0.1633 | 0.2633 | 0.2074 | 0.2123 |
| 81M2100356 | 0.1070 | 0.2492 | 0.1780 | 0.1828 |

Support means pixels above the fixed DAB threshold in reference-defined tissue; window MAE averages the absolute generated-versus-reference support-fraction difference for 32×32 reference-tissue windows (minimum 128 tissue pixels). Reference/output patch coordinates are used without independently verified cell alignment. The generated outputs remained visibly noisy; these DAB statistics are only image proxies and cannot determine biomarker correctness.

With the 300-step base, policy lr `1e-5` changed 45, 50 and 37 of 16,384 output pixels after PNG quantization (at most one RGB level). The exploratory lr `0.001` changed 13,108, 13,197 and 13,263 pixels, still at most one RGB level, and all three support-window errors increased. The final logged KL was 0.01642, and the rollback limit was 0.02; only logged steps can be inspected here. The result does not show useful reinforcement-learning improvements. Comparing the earlier and later flow objectives would change several factors and does not identify any one module's contribution.

## Engineering checks and remaining blockers

Ten CPU tests passed, including same-position PatchNCE gradients, no target in conditioning channels of the unconditional flow term, bounded reward behavior, three-channel action bounds, KL rollback and CLI smoke tests. These tests establish execution properties, not quality, localization, convergence, or generalization. No GPU or 30-hour timing test was available.

Most important: unpaired IHC distributions and shape-preserving losses do not identify the correct positive cells from HE. The region predictor remains trained within IHC and transferred to HE without validated cross-domain calibration; reward is not positivity ground truth. More data, a valid expression-supervision source, appropriate patient-independent splits, and a carefully designed held-out study are needed. The training and inference output at 300 steps is not adequate evidence for a full run.
