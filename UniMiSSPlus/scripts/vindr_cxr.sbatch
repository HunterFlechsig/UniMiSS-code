#!/bin/bash
#SBATCH -J vindr-cxr
#SBATCH -A grp_jliang12
#SBATCH -p general
#SBATCH -q public
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 7-0
#SBATCH -o /scratch/hflechsi/UniMiSS-code/UniMiSSPlus/scripts/logs/%x-%j.out
#SBATCH -e /scratch/hflechsi/UniMiSS-code/UniMiSSPlus/scripts/logs/%x-%j.err

# Full VinDr-CXR classification.
# Submit from a Sol login node:
#   sbatch /scratch/hflechsi/UniMiSS-code/UniMiSSPlus/scripts/vindr_cxr.sbatch

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

cd "${ROOT}/UniMiSSPlus/Downstream/2D/Cls"
python main.py \
  -train_list='dataset/VinDr-CXR/pe_global/train_pe_global_one.txt' \
  -test_list='dataset/VinDr-CXR/pe_global/test_pe_global_one.txt' \
  -GPU="${GPU}" \
  -BATCH_SIZE=32 \
  -EPOCH=30 \
  -deterministic=True \
  -LEARNING_RATE=0.0001 \
  -optimizer='AdamW' \
  -save_path='models/VinDr-CXR/' \
  -pre_train=True \
  -pre_train_path="${PRETRAIN}"
