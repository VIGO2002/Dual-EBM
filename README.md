# Dual-EBM: Class-Conditional Energy Decision Modeling for AI-Generated Image Detection

**Chenwei He (何晨炜)** · University of Electronic Science and Technology of China

**Paper:** *Class-Conditional Energy Decision Modeling for Generalizable AI-Generated Image Detection* — Chenwei He and Ke Qin. **IEEE ICASSP 2027 submission (under review)**.

Dual-EBM revisits the final decision layer of AI-generated image detectors. It models real and generated-image compatibility using **two independently parameterized energy heads** instead of a shared classifier. The objective combines negative-energy classification, an inter-head margin and local consistency under Gaussian perturbations in latent space.

## Method

```text
Input image
    │
CLIP ViT-L/14 backbone + SVD adaptation (rank ratio 0.25)
    │
Shared latent projection (1024 → 512)
    ├── Real energy head (512 → 256 → 1) ── E_real
    └── Fake energy head (512 → 256 → 1) ── E_fake
                                             
Logits:     [−E_real, −E_fake]
Inference:  predict the class with lower energy
Training:   cross entropy + 0.5 × margin loss + 0.1 × consistency loss
```

The perturbation branch operates **only during training**; inference requires one clean forward pass.

## Core implementation — start here

| Component | Source | What it demonstrates |
|---|---|---|
| **Dual energy heads** | [`dual_ebm/model.py`](dual_ebm/model.py) | Shared projection, independent real/fake energies, learnable perturbation |
| **Training objectives** | [`dual_ebm/losses.py`](dual_ebm/losses.py) | Paper Eqs. (3)–(5): classification, class-conditional margin, consistency |
| **SVD adaptation** | [`dual_ebm/svd.py`](dual_ebm/svd.py) | Attention-weight residual decomposition |
| **Optimization** | [`UFD_Benchmark/models/trainer.py`](UFD_Benchmark/models/trainer.py) | Trainable parameters, optimizer, gradient clipping |
| **Deepfake extension** | [`DeepfakeBench/training/detectors/effort_detector.py`](DeepfakeBench/training/detectors/effort_detector.py) | Separate deepfake detection adaptation |

## Reported results

| Benchmark | Training source | Manuscript result |
|---|---|---|
| UniversalFakeDetect | ProGAN | **99.53% mAP / 95.11% Worst-Case AP** |
| GenImage | Stable Diffusion v1.4 | **93.2% mean accuracy** |
| AIGIBench | ProGAN; no target-specific retraining | **79.68% AP** |

These results are **reported in the submitted paper**, not newly reproduced from this reorganized code. The additional Deepfake detection experiments were not included in the paper due to space limitations.

## Repository layout

```text
Dual-EBM/
├── dual_ebm/                # Main algorithm: energy heads, losses, SVD
├── configs/                 # Manuscript hyperparameters
├── UFD_Benchmark/           # UFD training and evaluation
├── DeepfakeBench/           # Additional deepfake research adaptation
├── tests/                   # Synthetic model tests
├── docs/                    # Reproducibility details
├── requirements.txt
└── THIRD_PARTY_NOTICES.md
```

## Setup and usage

Use Python 3.10+ and a compatible PyTorch/CUDA installation:

```bash
git clone https://github.com/VIGO2002/Dual-EBM.git
cd Dual-EBM
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

UFD training uses the author's local dataset root at `/root/autodl-tmp/datasets/CNNDetection`. The script requires an **independently held-out ProGAN validation directory** containing `0_real/` and `1_fake/`:

```bash
bash UFD_Benchmark/script/train_svd.sh /absolute/path/to/held_out_progan
```

Evaluation requires a trained checkpoint:

```bash
bash UFD_Benchmark/script/test.sh /path/to/model_epoch_N.pth
```

The local CLIP ViT-L/14 weights at `/root/autodl-tmp/pretrained_models/clip-vit-large-patch14` are preferred when present; otherwise a public pretrained model identifier is used. Datasets and checkpoints are not distributed here.

**Reproduction caveat:** The manuscript does not give every numerical phase value for its step-wise LambdaLR schedule. These settings remain **provisional** in the reorganized code, which has **not** undergone full original-checkpoint or end-to-end numerical equivalence testing. See [reproducibility notes](docs/reproducibility.md).

## Authorship and upstream credits

**Dual-EBM research, algorithm implementation, experiments and Deepfake adaptation:** **Chenwei He (何晨炜)**.

Benchmark infrastructure incorporates adapted components from [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) and [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench). Upstream framework authorship is distinct from the novel Dual-EBM contribution. Their original license/attribution terms remain applicable; see [third-party notices](THIRD_PARTY_NOTICES.md).
