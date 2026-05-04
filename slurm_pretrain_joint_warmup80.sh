#!/bin/bash
#SBATCH --job-name=pretrain_joint_pw80
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_warmup80/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_warmup80/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# ===================================================================
# Joint base con warmup más largo para la rama ProtoNet.
#
# Hipótesis:
#   retrasar la entrada fuerte de la pérdida episódica puede dejar que JEPA
#   estructure mejor el embedding antes de forzarlo hacia la métrica few-shot.
# ===================================================================

# -- Entorno
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

mkdir -p logs/gadf_soil_nir_joint_warmup80

echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "Config:        configs/gadf_soil_nir_joint.yaml"
echo "GPU:           cuda:0"
echo "Proto warmup:  80"
echo "Lambda:        0.025"
echo "Epochs:        400"
echo "=========================================="

python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint.yaml \
    --devices cuda:0 \
    --folder logs/gadf_soil_nir_joint_warmup80 \
    --tag gadf_joint_warmup80 \
    --epochs 400 \
    --meta_batch_size 8 \
    --lambda_proto 0.025 \
    --proto_warmup 80 \
    --wandb_name "joint_noproj_lp0.025_smoothl1_warmup80_ep400"
