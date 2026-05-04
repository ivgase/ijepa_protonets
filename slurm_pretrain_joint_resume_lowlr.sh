#!/bin/bash
#SBATCH --job-name=pretrain_joint_resume_lowlr
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_resume_lowlr_base600/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_resume_lowlr_base600/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# ===================================================================
# Resume conservador desde el mejor joint actual (ep600).
#
# Objetivo:
#   extender 600 -> 800 épocas con una cola más suave que el schedule
#   original, aislando el checkpoint de arranque en un directorio nuevo
#   para no sobrescribir el latest que ya estamos reutilizando en otros jobs.
#
# Ajustes clave:
#   - lr:       5e-4
#   - start_lr: 5e-5
#   - final_lr: 5e-7
#   - warmup:   10
#   - epochs:   800
# ===================================================================

# -- Entorno
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

BASE_CKPT_DIR="/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint/gadf_joint_20260414_094939"
BASE_CKPT="${BASE_CKPT_DIR}/gadf_joint-latest.pth.tar"
RESUME_DIR="/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_resume_lowlr_base600"
RESUME_CKPT="${RESUME_DIR}/gadf_joint_base600.pth.tar"

mkdir -p "${RESUME_DIR}"

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "Copiando checkpoint base de 600 epocas a ${RESUME_CKPT}"
    cp "${BASE_CKPT}" "${RESUME_CKPT}"
else
    echo "Checkpoint base ya presente en ${RESUME_CKPT}"
fi

if [ -f "${BASE_CKPT_DIR}/params-joint.yaml" ] && [ ! -f "${RESUME_DIR}/params-joint.base600.yaml" ]; then
    cp "${BASE_CKPT_DIR}/params-joint.yaml" "${RESUME_DIR}/params-joint.base600.yaml"
fi

echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Config:      configs/gadf_soil_nir_joint_resume_lowlr.yaml"
echo "GPU:         cuda:0"
echo "Checkpoint:  ${RESUME_CKPT}"
echo "Base model:  ${BASE_CKPT}"
echo "Plan:        ep600 -> ep800 con LR conservador"
echo "W&B name:    joint_noproj_lp0.025_smoothl1_resume800_lowlr"
echo "=========================================="

python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint_resume_lowlr.yaml \
    --devices cuda:0
