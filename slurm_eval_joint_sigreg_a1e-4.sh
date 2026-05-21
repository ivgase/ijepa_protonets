#!/bin/bash
#SBATCH --job-name=eval_sigreg_a1e-4
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/eval_sigreg_a1e-4/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/eval_sigreg_a1e-4/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

# ===================================================================
# Eval ProtoNet zero-shot en TEST — sigreg a1e-4, cap10, wu50 (ep1200)
#
# Evalúa los dos mejores checkpoints según val probe durante entrenamiento:
#   - ep1000: val R²=0.4761  ← mejor
#   - ep1150: val R²=0.4759  ← segundo
#
# Modelo: joint savgol_d1 + transformer + Weak-SIGReg (alpha=1e-4)
# Job de pretraining: 136551
# Checkpoint dir: logs/gadf_soil_nir_joint_savgol_d1_transformer_sigreg/
#                 gadf_joint_savgol_d1_transformer_sigreg_ep1200_sigreg_a1e-4_cap10_wu50/
#
# Config: use_projection=true, projection_type=transformer → usa
#         main_eval_joint_protonet.py (mismo que slurm_eval_joint_savgol_d1_transformer.sh)
#
# Objetivo: obtener el R² en test del mejor modelo pre-entrenado actual
#           y comparar con R²=0.504 (scratch vit_1d_aligned) y 0.451
#           (joint_transformer_ep800, job 131424).
#
# Usage:
#   ./launch.sh slurm_eval_joint_sigreg_a1e-4.sh \
#       "<motivo del experimento>"
# ===================================================================

set +u
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
set -euo pipefail
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

CKPT_DIR="logs/gadf_soil_nir_joint_savgol_d1_transformer_sigreg/gadf_joint_savgol_d1_transformer_sigreg_ep1200_sigreg_a1e-4_cap10_wu50"
CONFIG="${CKPT_DIR}/params-joint.yaml"
DATA_PATH="data/SoilDataset_NIR_gadf2d_savgol_d1"
K_SPT=25
K_QRY=25
BATCH_SIZE=64

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  ${CONFIG}"
echo "Data:    ${DATA_PATH}"
echo "Split:   test  |  k_spt=${K_SPT}  |  k_qry=${K_QRY}"
echo "=========================================="

for EP in 1000 1150; do
    CKPT="${CKPT_DIR}/gadf_joint_savgol_d1_transformer_sigreg-ep${EP}.pth.tar"
    SAVE="results/eval_joint_proto/sigreg_a1e-4_ep${EP}_test_k${K_SPT}/"

    echo ""
    echo "--- Evaluando ep${EP} ---"
    echo "Ckpt: ${CKPT}"
    echo "Save: ${SAVE}"

    /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/bin/python main_eval_joint_protonet.py \
        --checkpoint  "${CKPT}" \
        --config      "${CONFIG}" \
        --split       test \
        --data_path   "${DATA_PATH}" \
        --k_spt       "${K_SPT}" \
        --k_qry       "${K_QRY}" \
        --batch_size  "${BATCH_SIZE}" \
        --save_path   "${SAVE}" \
        --device      cuda

    echo "Resultados ep${EP} guardados en: ${SAVE}"
done

echo ""
echo "=========================================="
echo "Eval finalizado."
echo "  ep1000 → results/eval_joint_proto/sigreg_a1e-4_ep1000_test_k${K_SPT}/"
echo "  ep1150 → results/eval_joint_proto/sigreg_a1e-4_ep1150_test_k${K_SPT}/"
echo "Comparar con:"
echo "  - R²=0.451 (joint_transformer_ep800, job 131424)"
echo "  - R²=0.504 (scratch vit_1d_aligned lr=2e-3, job 136574)"
echo "  - R²=0.490 (baseline ResNet18+ProtoNets)"
echo "=========================================="
