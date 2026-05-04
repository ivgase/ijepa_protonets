#!/bin/bash
#SBATCH --job-name=proto_scratch_5e-5
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
# ijepa_spectra2d — ProtoNet from scratch, lr=5e-5
#
# Objetivo: comprobar si lr=5e-5 supera al mejor del sweep anterior
# (lr=1e-4, test R²=0.411). La tendencia monótona del sweep sugería
# que el fondo de la curva no se había alcanzado todavía.
#
# start_lr = final_lr = lr/100 = 5e-7 (misma regla que sweep anterior)
# 1000 epochs, misma arquitectura y config que el sweep.
#
# Usage:
#   ./launch.sh slurm_protonet_scratch_lr5e-5.sh "probar lr=5e-5 en protonet scratch, mejor del sweep fue lr=1e-4 (R²=0.411)"
# ===================================================================

set -eo pipefail

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

mkdir -p logs/gadf_soil_nir_protonet_scratch

RUN_DIR="logs/gadf_soil_nir_protonet_scratch/lr5e-5_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_protonet_scratch.yaml"
echo "lr=5e-5  start_lr=5e-7  final_lr=5e-7"
echo "Epochs:   1000"
echo "Output:   ${RUN_DIR}/"
echo "=========================================="

python main_protonet_only.py \
    --fname configs/gadf_soil_nir_protonet_scratch.yaml \
    --devices cuda:0 \
    --lr "5e-5" \
    --start_lr "5e-7" \
    --final_lr "5e-7" \
    --epochs 1000 \
    --folder "${RUN_DIR}" \
    --tag "lr5e-5" \
    --wandb_name "proto_scratch_lr5e-5_${SLURM_JOB_ID}"

echo ""
echo "=========================================="
echo "Run completado."
echo "Para ver el R² de test:"
echo "  grep 'AVERAGE' ${RUN_DIR}/lr5e-5_*/eval_test/results_per_task.csv"
echo "=========================================="
