#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT/UniversalFakeDetect_Benchmark - 副本"
if [ "$#" -ne 1 ]; then
    echo "Usage: bash script/train_svd.sh /absolute/path/to/held_out_progan" >&2
    echo "The directory must contain 0_real and 1_fake classes, held out from training." >&2
    exit 2
fi
VAL_ROOT="$1"
python train.py \
  --name effort_universal_repro \
  --arch CLIP:ViT-L/14_svd --fix_backbone --use_svd \
  --data_mode wang2020 \
  --wang2020_data_path /root/autodl-tmp/datasets/CNNDetection \
  --val_root "$VAL_ROOT" \
  --batch_size 32 --niter 15 --lr 0.0002 --weight_decay 0 \
  --warmup_steps 0 --grad_clip 5 --noise_std 0.02 \
  --energy_margin 5 --lambda_margin 0.5 --lambda_smooth 0.1 \
  --jpg_prob 0.5 --blur_prob 0.5 --data_aug \
  --loss_freq 10 --save_epoch_freq 1
