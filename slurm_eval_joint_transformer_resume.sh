#!/bin/bash

#SBATCH --job-name=eval_transformer_resume
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/eval_joint_proto_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/eval_joint_proto_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

# ===================================================================
# Eval ProtoNet zero-shot — joint transformer resume (ep750)
# Mejor probe val R²=0.5200 (ep750). Sin Savitzky-Golay.
# ===================================================================

set +u
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
set -euo pipefail
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

CKPT="logs/gadf_soil_nir_joint_transformer_resume/gadf_joint_transformer_resume_ep400base/gadf_joint_transformer_resume-ep750.pth.tar"
CONFIG="logs/gadf_soil_nir_joint_transformer_resume/gadf_joint_transformer_resume_ep400base/params-joint.yaml"
DATA_PATH="data/SoilDataset_NIR_gadf2d"
SPLIT="test"
K_SPT=25
K_QRY=25
BATCH_SIZE=64
SAVE="results/eval_joint_proto/gadf_joint_transformer_resume_ep750_${SPLIT}_k${K_SPT}/"

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Ckpt:    $CKPT"
echo "Split:   $SPLIT  |  k_spt=$K_SPT  |  k_qry=$K_QRY"
echo "Save:    $SAVE"
echo "=========================================="

/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/bin/python main_eval_joint_protonet.py \
    --checkpoint  "${CKPT}" \
    --config      "${CONFIG}" \
    --split       "${SPLIT}" \
    --data_path   "${DATA_PATH}" \
    --k_spt       "${K_SPT}" \
    --k_qry       "${K_QRY}" \
    --batch_size  "${BATCH_SIZE}" \
    --save_path   "${SAVE}" \
    --device      cuda

echo "=========================================="
echo "Eval finalizado. Resultados en: ${SAVE}"
echo "=========================================="
