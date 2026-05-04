#!/bin/bash
#SBATCH --job-name=proto_scratch_savgol_d1
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
# ijepa_spectra2d — ProtoNet from scratch, lr=1e-4, savgol_d1 (deriv=1)
#
# Objetivo: baseline ProtoNet from scratch con preprocesamiento
# Savitzky-Golay primera derivada (window=15, poly=2, deriv=1).
# lr=1e-4 fue el mejor del sweep anterior (R²=0.411 sin preprocesar).
#
# Usage:
#   ./launch.sh slurm_protonet_scratch_savgol_d1.sh "protonet scratch lr=1e-4 savgol deriv=1"
# ===================================================================

set -eo pipefail

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

mkdir -p logs/gadf_soil_nir_protonet_scratch

RUN_DIR="logs/gadf_soil_nir_protonet_scratch/savgol_d1_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Variante: savgol_d1 (deriv=1)"
echo "lr=1e-4  start_lr=1e-6  final_lr=1e-6  epochs=1000"
echo "Output:   ${RUN_DIR}/"
echo "=========================================="

python main_protonet_only.py \
    --fname configs/gadf_soil_nir_protonet_scratch.yaml \
    --devices cuda:0 \
    --lr 1e-4 \
    --start_lr 1e-6 \
    --final_lr 1e-6 \
    --epochs 1000 \
    --folder "${RUN_DIR}" \
    --tag "savgol_d1" \
    --proto_data_path "data/SoilDataset_NIR_gadf2d_savgol_d1" \
    --probe_data_path "data/SoilDataset_NIR_gadf2d_savgol_d1" \
    --gadf_norm_stats "data/gadf_norm_stats_v2_savgol_d1.json" \
    --wandb_name "proto_scratch_savgol_d1_${SLURM_JOB_ID}"

echo ""
echo "Run completado."
echo "Para ver el R² de test:"
echo "  grep 'AVERAGE' ${RUN_DIR}/savgol_d1_*/eval_test/results_per_task.csv"
