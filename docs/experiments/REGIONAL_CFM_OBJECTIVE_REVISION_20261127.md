# Regional CFM objective revision — 2026-11-27

This code revision removes the assumption that filename-paired serial sections are pixel-registered.

## Changes

- Generator loss count remains three: regional conditional flow matching, PatchNCE, and adversarial loss.
- CFM keeps the paired H&E condition and paired IHC endpoint, but replaces pixelwise velocity MSE with four-phase, valid-area-weighted regional velocity MSE. The default `flow_region` is 64 px.
- The marker-conditioned PatchGAN now receives IHC RGB only. It no longer uses paired H&E/IHC coordinates as a realism signal.
- PPO keeps its paired-reference DAB support/amount reward at `reward_region=128` px over four grid phases.
- Checkpoint format is 9. Earlier checkpoints use a different objective and cannot be resumed for this run.

## Interpretation

The change makes small local section shifts less dominant in training. It does not register images, correct large/nonrigid displacement, or provide cell-level expression labels. The model is trained with paired regional expression supervision; claims must stay at that scale. Existing format-8 short-run metrics are implementation smoke results for the previous objective and are not evidence for format 9.

## Training entry points

Prepare the paired cache, train a fresh base flow checkpoint, then start joint PPO training from that checkpoint. Do not use an older checkpoint with `--resume` or as `train-rl --checkpoint`.
