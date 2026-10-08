#!/bin/bash
# One epoch of VinDr-CXR classification (including its test-list scoring),
# then one epoch of RICORD classification.
#
# On a Sol login node, start a short session first:
#   interactive -A grp_jliang12 -p htc -q public -G 1 -c 8 --mem=32G -t 0-4
# Then:
#   bash /scratch/hflechsi/UniMiSS-code/UniMiSSPlus/scripts/smoke_vindr_ricord.sh

set -euo pipefail

ROOT=/scratch/hflechsi/UniMiSS-code
PRETRAIN="${ROOT}/UniMiSSPlus/snapshots/UniMissPlus/UniMissPlus.pth"

host=$(hostname -s)
case "${host}" in
  sol-login*)
    echo "This host is ${host}. Open a compute session first:" >&2
    echo "  interactive -A grp_jliang12 -p htc -q public -G 1 -c 8 --mem=32G -t 0-4" >&2
    exit 1
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  module load mamba/latest
fi
# conda's own startup reads variables that are unset under `set -u`.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate UniMiSSPlus
set -u

GPU="${CUDA_VISIBLE_DEVICES:-0}"
GPU="${GPU%%,*}"

echo "=== VinDr-CXR smoke $(date) ==="
cd "${ROOT}/UniMiSSPlus/Downstream/2D/Cls"
python main.py \
  -train_list='dataset/VinDr-CXR/pe_global/train_pe_global_one.txt' \
  -test_list='dataset/VinDr-CXR/pe_global/test_pe_global_one.txt' \
  -GPU="${GPU}" \
  -BATCH_SIZE=32 \
  -EPOCH=1 \
  -deterministic=True \
  -LEARNING_RATE=0.0001 \
  -optimizer='AdamW' \
  -save_path='models/smoke-vindr/' \
  -pre_train=True \
  -pre_train_path="${PRETRAIN}"

echo "=== RICORD smoke $(date) ==="
cd "${ROOT}/UniMiSSPlus/Downstream/3D/RICORD"
python train.py \
  -train_list='lists/RICORD_train.txt' \
  -val_list='lists/RICORD_val.txt' \
  -GPU="${GPU}" \
  -NUM_CLASSES=2 \
  -BATCH_SIZE=8 \
  -EPOCH=1 \
  -TRAIN_NUM=512 \
  -LEARNING_RATE=0.00001 \
  -optimizer='AdamW' \
  -save_path='models/smoke-ricord/' \
  -pre_train=True \
  -pre_train_path="${PRETRAIN}"

echo "=== smoke run finished $(date) ==="
