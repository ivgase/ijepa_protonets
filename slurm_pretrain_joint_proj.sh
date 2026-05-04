#!/bin/bash
#SBATCH --job-name=pretrain_joint_proj
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_proj/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_proj/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# ── Entorno ──────────────────────────────────────────────────────────────────
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

# ── Directorio de logs ────────────────────────────────────────────────────────
mkdir -p logs/gadf_soil_nir_joint_proj

# ── Info del job ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_joint_proj.yaml"
echo "GPU:      cuda:0"
echo "Head:     ProjectionNet(768 -> 256 -> 128)"
echo "=========================================="

# ── Joint Pretraining (I-JEPA + ProtoNet con cabeza de proyección) ────────────
# ProjectionNet: Linear(768,256) -> LayerNorm -> ReLU -> Linear(256,128)
# La cabeza se entrena junto con el encoder en la fase ProtoNet (lambda_proto=0.05).
python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint_proj.yaml \
    --devices cuda:0 \
    --meta_batch_size 8 \
    --epochs 400 \
    --lambda_proto 0.025 \
    --wandb_name "joint_proj_mb8_lp0.025_smoothl1"
