# Reproducibility notes

**Author of Dual-EBM: Chenwei He (何晨炜).** Manuscript: *Class-Conditional
Energy Decision Modeling for Generalizable AI-Generated Image Detection*,
IEEE ICASSP 2027 submission **under review**.

The paper specifies UFD training on ProGAN for 15 epochs, AdamW at 2e-4
with betas (0.9, 0.999), batch 32, no warmup, weight decay 0, gradient
clipping 5, SVD rank ratio 0.25, energy margin 5, margin weight 0.5,
consistency weight 0.1, learnable sigma initialized at 0.02, clamped
to [0.001, 0.05], JPEG and blur each at probability 0.5.

**Known unresolved parameter:** The paper describes an epoch-1 linear
decay followed by step-wise cosine restarts using LambdaLR, but does
not give the first-stage endpoint, restart interval, or floor.
The `--linear_end`, `--restart_epochs`, and `--minimum_lr_factor`
default values are provisional. They must be matched to the historical
training log before claiming strict numerical reproduction.

**Data selection:** UFD train=ProGAN; validation=held-out ProGAN only;
test=unseen generators. GenImage training=SDv1.4, not ProGAN. Reuse of
a ProGAN checkpoint does not replicate the published GenImage result.
Dataset/CLIP absolute local paths are intentionally retained.

**Deepfake extension authorship:** Chenwei He implemented and modified the
Dual-EBM Deepfake detector and experimental workflow on top of DeepfakeBench.
Original framework code retains its original attribution under the upstream
license; the Deepfake extension is not among the evaluations described in
the conference manuscript.

**Legacy checkpoint validation:** The new SVD module preserves the
old attention parameter names and shapes. Before using old checkpoints,
compare a fixed-batch logit tensor with the original hook-based code,
then verify a strict state_dict load. Existing weights were produced
by historical scripts; updating the training code does not retroactively
change those experiments.

## Source-only validation guard

UFD training requires an explicit `--val_root` containing exactly
`0_real` and `1_fake` class folders. The caller must confirm that
the samples were held out from training and originate only from ProGAN.
The example `/root/autodl-tmp/datasets/CNNDetection/val/progan` is a
suggested local directory layout, not a verified existing path.

## Deepfake auxiliary training

The separate DeepfakeBench scripts retain their own historical optimizer
and SVD settings. Their cosine scheduler is advanced per training epoch,
not as UFD's paper LambdaLR. They still require compatible third-party
dependencies and a CUDA GPU. Their published provenance is described in
`DeepfakeBench/README.md`.
