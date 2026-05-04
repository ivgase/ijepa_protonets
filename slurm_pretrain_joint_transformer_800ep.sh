#!/bin/bash
#SBATCH --job-name=joint_transformer_800ep
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_transformer/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_transformer/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

# ===================================================================
# Joint transformer — 800 épocas continuas, sin reanudaciones.
# Un único cosine schedule 0→800 para evitar el salto de LR que
# producía la versión en etapas (slurm_pretrain_joint_transformer_resume.sh).
# ===================================================================

set -eo pipefail

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

RUN_ID="ep800_continuous"
BASE_LOG="$(pwd)/logs/gadf_soil_nir_joint_transformer"
mkdir -p "${BASE_LOG}"

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  configs/gadf_soil_nir_joint_transformer.yaml"
echo "Flujo:   ep0→800, coseno único, sin resume"
echo "=========================================="

python main_joint_pretrain.py \
    --fname   configs/gadf_soil_nir_joint_transformer.yaml \
    --devices cuda:0 \
    --run_id  "${RUN_ID}" \
    --epochs        800 \
    --final_lr      5e-7 \
    --lambda_proto  0.025 \
    --wandb_name    "joint_transformer_ep800_continuous"

echo "=========================================="
echo "Entrenamiento finalizado. Job $SLURM_JOB_ID done."
echo "=========================================="
