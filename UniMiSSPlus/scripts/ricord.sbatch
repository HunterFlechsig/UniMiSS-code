#!/bin/bash
#SBATCH -J ricord
#SBATCH -A grp_jliang12
#SBATCH -p private
#SBATCH -q private
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 7-0
#SBATCH -o /scratch/hflechsi/UniMiSS-code/UniMiSSPlus/scripts/logs/%x-%j.out
#SBATCH -e /scratch/hflechsi/UniMiSS-code/UniMiSSPlus/scripts/logs/%x-%j.err

# Full RICORD classification, matching Downstream/3D/RICORD/readme.md.
# Submit from a Sol login node:
#   sbatch /scratch/hflechsi/UniMiSS-code/UniMiSSPlus/scripts/ricord.sbatch

set -euo pipefail

ROOT=/scratch/hflechsi/UniMiSS-code
PRETRAIN="${ROOT}/UniMiSSPlus/snapshots/UniMissPlus/UniMissPlus.pth"

if ! command -v conda >/dev/null 2>&1; then
  module load mamba/latest
fi
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate UniMiSSPlus
set -u

GPU="${CUDA_VISIBLE_DEVICES:-0}"
GPU="${GPU%%,*}"

cd "${ROOT}/UniMiSSPlus/Downstream/3D/RICORD"
python train.py \
  -train_list='lists/RICORD_train.txt' \
  -val_list='lists/RICORD_val.txt' \
  -GPU="${GPU}" \
  -NUM_CLASSES=2 \
  -BATCH_SIZE=8 \
  -EPOCH=200 \
  -TRAIN_NUM=512 \
  -LEARNING_RATE=0.00001 \
  -optimizer='AdamW' \
  -save_path='models/' \
  -pre_train=True \
  -pre_train_path="${PRETRAIN}"
