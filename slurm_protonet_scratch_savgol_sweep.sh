#!/bin/bash
#SBATCH --job-name=proto_scratch_savgol
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_protonet_scratch/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_protonet_scratch/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

# ===================================================================
# ijepa_spectra2d — ProtoNet from scratch, lr=1e-4, variantes savgol
#
# Objetivo: comparar el efecto del preprocesamiento Savitzky-Golay
# en el baseline ProtoNet from scratch.  lr=1e-4 fue el mejor del
# sweep anterior (R²=0.411 con datos sin preprocesar).
#
# Variantes:
#   1. savgol      — SG(window=15, poly=2, deriv=0)
#   2. savgol_d1   — SG(window=15, poly=2, deriv=1)
#
# lr=1e-4, start_lr=1e-6, final_lr=1e-6 (misma regla que sweep).
# 1000 epochs por run.
#
# Usage:
#   ./launch.sh slurm_protonet_scratch_savgol_sweep.sh "protonet scratch lr=1e-4, variantes savgol y savgol_d1"
# ===================================================================

set -eo pipefail

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

mkdir -p logs/gadf_soil_nir_protonet_scratch

SWEEP_DIR="logs/gadf_soil_nir_protonet_scratch/savgol_sweep_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR"

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_protonet_scratch.yaml"
echo "GPU:      cuda:0"
echo "lr=1e-4  start_lr=1e-6  final_lr=1e-6"
echo "Epochs:   1000 por run"
echo "Variantes: savgol, savgol_d1"
echo "Output:   ${SWEEP_DIR}/"
echo "=========================================="

# ---------------------------------------------------------------------------
# Run 1: savgol (deriv=0)
# ---------------------------------------------------------------------------
echo ""
echo "================================================"
echo "[RUN 1/2] savgol (deriv=0)"
echo "================================================"

python main_protonet_only.py \
    --fname configs/gadf_soil_nir_protonet_scratch.yaml \
    --devices cuda:0 \
    --lr 1e-4 \
    --start_lr 1e-6 \
    --final_lr 1e-6 \
    --epochs 1000 \
    --folder "${SWEEP_DIR}" \
    --tag "savgol" \
    --proto_data_path "data/SoilDataset_NIR_gadf2d_savgol" \
    --probe_data_path "data/SoilDataset_NIR_gadf2d_savgol" \
    --gadf_norm_stats "data/gadf_norm_stats_v2_savgol.json" \
    --wandb_name "proto_scratch_savgol_${SLURM_JOB_ID}"

# ---------------------------------------------------------------------------
# Run 2: savgol_d1 (deriv=1)
# ---------------------------------------------------------------------------
echo ""
echo "================================================"
echo "[RUN 2/2] savgol_d1 (deriv=1)"
echo "================================================"

python main_protonet_only.py \
    --fname configs/gadf_soil_nir_protonet_scratch.yaml \
    --devices cuda:0 \
    --lr 1e-4 \
    --start_lr 1e-6 \
    --final_lr 1e-6 \
    --epochs 1000 \
    --folder "${SWEEP_DIR}" \
    --tag "savgol_d1" \
    --proto_data_path "data/SoilDataset_NIR_gadf2d_savgol_d1" \
    --probe_data_path "data/SoilDataset_NIR_gadf2d_savgol_d1" \
    --gadf_norm_stats "data/gadf_norm_stats_v2_savgol_d1.json" \
    --wandb_name "proto_scratch_savgol_d1_${SLURM_JOB_ID}"

echo ""
echo "=========================================="
echo "Todos los runs completados."
echo "Para comparar R² entre variantes:"
echo "  grep 'AVERAGE' ${SWEEP_DIR}/savgol*/eval_test/results_per_task.csv"
echo "=========================================="
