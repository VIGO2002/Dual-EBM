#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT/UniversalFakeDetect_Benchmark - 副本"
if [ "$#" -ne 1 ]; then
    echo "Usage: bash script/test.sh /path/to/model_epoch_N.pth" >&2
    exit 2
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python validate.py \
  --arch CLIP:ViT-L/14_svd --ckpt "$1" \
  --result_folder results/paper_eval --use_svd --fix_backbone
