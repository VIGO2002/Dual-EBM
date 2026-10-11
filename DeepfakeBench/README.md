# Dual-EBM — Deepfake evaluation extension

**Dual-EBM project author and Deepfake extension developer:** **Chenwei He (何晨炜)**

This directory holds Chenwei He's Deepfake detection/evaluation adaptation
of the Dual-EBM class-conditional energy method. It includes model-level
changes such as separate real/fake energy branches, an inter-head margin,
latent perturbation consistency, experiment configurations, and adaptations
of the benchmark training/evaluation workflow.

**Publication scope:** The ICASSP 2027 submission describes the UFD,
GenImage, and AIGIBench experiments. Its page limit meant that this
Deepfake evaluation extension was not included. Results for this extension
must not be presented as results reported in that manuscript.

## Upstream attribution and licensing

The training, dataset loading, augmentation, metrics, and evaluation
infrastructure comes from the open-source
[DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) project.
The original framework is attributed to its upstream contributors,
including **Zhiyuan Yan**. The upstream project identifies
**The Chinese University of Hong Kong, Shenzhen (CUHK(SZ))**
in its 2023 copyright notice.

The upstream code is distributed under
[CC BY-NC 4.0](https://github.com/SCLBD/DeepfakeBench/blob/main/LICENSE),
with exceptions for third-party components described in that license.
This project retains source-level upstream authorship notices and
identifies Chenwei He's subsequent adaptations. Attribution to an
upstream framework does **not** make its authors co-authors of Dual-EBM.

## Structure

- `training/detectors/effort_detector.py` — the adapted Dual-EBM
  class-conditional detector implementation
- `training/trainer/trainer.py` — framework trainer and adapted training workflow
- `training/config/detector/effort.yaml` — detector/hyperparameter settings
- `analysis/` — diagnostic scripts
- `preprocessing/` — supporting data preparation and dataset indices

## Reproducibility status

The Deepfake workflow is a separate experiment from the paper's main
UFD/GenImage configurations. Legacy hyperparameters and local paths
are preserved where possible. Its benchmark results have **not**
been revalidated against the reorganized core module in this branch.

Do not equate this directory with a clean-room reimplementation of
all DeepfakeBench infrastructure. It is a research adaptation with
explicit upstream provenance.
