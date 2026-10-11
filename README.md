# Dual-EBM: Class-Conditional Energy Decision Modeling

**Author:** Chenwei He (何晨炜) · University of Electronic Science and Technology of China

**Paper:** *Class-Conditional Energy Decision Modeling for Generalizable
AI-Generated Image Detection*, Chenwei He and Ke Qin.
IEEE ICASSP 2027 submission (**under review**, not accepted).

Dual-EBM learns independently parameterized real/fake energy functions
over a shared CLIP latent representation. Training combines
classification on negative energies, an inter-head margin and
consistency under local Gaussian perturbations. Inference uses one
clean forward pass and predicts the lower-energy class.

## Paper-reported results

| Benchmark | Training source | Reported metric |
|---|---|---|
| UniversalFakeDetect | ProGAN | mAP **99.53%**, worst AP **95.11%** |
| GenImage | SDv1.4 | mAcc **93.2%** |
| AIGIBench (no target retraining) | ProGAN | AP **79.68%** |

These are **paper-reported** numbers. The cleanup branch has not yet
been retrained or verified to reproduce them.

## Architecture

```text
Image → pretrained CLIP ViT-L/14 → shared 1024→512 latent projection
                                   ├─ E_real : 512→256→1
                                   └─ E_fake : 512→256→1
Clean logits = [-E_real, -E_fake]
Loss = CE + 0.5 × conditional margin (m=5) + 0.1 × consistency
```

## Repository guide

- `dual_ebm/` — original decision module, losses, and SVD adaptation
- `configs/` — paper-stated hyperparameters
- `UniversalFakeDetect_Benchmark - 副本/` — historical UFD experiments
- `DeepfakeBench/` — **Chenwei He's Dual-EBM deepfake extension**, built on the
  upstream DeepfakeBench framework; see [module notes](DeepfakeBench/README.md)
- `tests/` — synthetic correctness checks without downloading datasets
- `docs/reproducibility.md` — protocol provenance and open caveats

## Setup and UFD training

Install a CUDA-compatible version of PyTorch, then:

```bash
git clone https://github.com/VIGO2002/Dual-EBM.git
cd Dual-EBM
pip install -r requirements.txt
bash 'UniversalFakeDetect_Benchmark - 副本/script/train_svd.sh' \
    /root/autodl-tmp/datasets/CNNDetection/val
```

The held-out validation folder **must be ProGAN source data** with
`0_real` and `1_fake` directories. Never select a checkpoint on
unseen-generator test data. Paths to the author's local datasets and
pretrained weights are deliberately unchanged.

```bash
bash 'UniversalFakeDetect_Benchmark - 副本/script/test.sh' \
    /path/to/model_epoch_N.pth
python -m unittest discover -s tests -v
```

**Reproduction caveat:** The manuscript specifies linear decay followed
by cosine restarts, but omits numerical phase endpoints. Current defaults
for these undocumented parameters are provisional. See
[reproducibility notes](docs/reproducibility.md).

## Credits / provenance

Original Dual-EBM contribution: **Chenwei He**.
**Deepfake extension and adaptation of the detector/training workflow:** **Chenwei He**.
The Deepfake evaluation work is distinct from the experiments reported in the
five-page ICASSP manuscript.
The benchmark adaptations incorporate code from
[UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect)
and [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench).
Their original attribution and applicable licenses should be preserved;
upstream contributors are not the authors of the new Dual-EBM method.
