#!/bin/bash
#SBATCH --job-name=proto_scratch_smoke
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
# ijepa_spectra2d — Smoke test ProtoNet from scratch (ViT-base, GADF)
#
# Objetivo: validar que el modo load_checkpoint=false funciona
# correctamente antes de lanzar el sweep largo.
#
# Verificar en el log que aparece:
#   'No checkpoint loaded — training from random weights'
#   'ProtoNet sampler: N train tasks'  (N ≈ 120)
# Y en W&B que train/loss_proto baja desde ≈1.0 en los primeros epochs.
#
# Tiempo estimado: ~30 min (50 epochs)
#
# Usage:
#   ./launch.sh slurm_protonet_scratch_smoke.sh "validar modo from-scratch"
# ===================================================================

set -eo pipefail

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

mkdir -p logs/gadf_soil_nir_protonet_scratch

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_protonet_scratch.yaml"
echo "GPU:      cuda:0"
echo "Objetivo: Smoke test ProtoNet from scratch (50 epochs)"
echo "Arquitectura: vit_base, patch=16, GADF 224x224"
echo "NOTA: NO se carga ningún checkpoint (random init)"
echo "=========================================="

python main_protonet_only.py \
    --fname configs/gadf_soil_nir_protonet_scratch.yaml \
    --devices cuda:0 \
    --epochs 50 \
    --tag smoke \
    --probe_freq 10 \
    --wandb_name "proto_scratch_smoke_${SLURM_JOB_ID}"
