#!/bin/bash

#SBATCH --job-name=eval_joint_proto
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/eval_joint_proto_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/eval_joint_proto_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

# ===================================================================
# ijepa_spectra2d — ProtoNet zero-shot eval con checkpoint joint
#
# Carga target_encoder + proto_projection del checkpoint y evalúa
# directamente sobre el split test sin ningún paso de entrenamiento.
#
# Usa los mismos fixed_val splits y métricas que main_finetune_gadf2d.py
# para que los resultados sean directamente comparables.
#
# Usage:
#   sbatch slurm_eval_joint_protonet.sh
# ===================================================================

# conda activation uses unset vars internally — protect with set +u
set +u
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
set -euo pipefail
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d
mkdir -p logs

# ------ Config ------
DATA_PATH="data/SoilDataset_NIR_gadf2d"
SPLIT="test"
K_SPT=25
K_QRY=25
BATCH_SIZE=64

# Lista de (checkpoint, config) a evaluar.
# El config guardado junto al checkpoint (params-joint.yaml) refleja exactamente
# los hiperparámetros del run, incluyendo la arquitectura de proto_projection.
RUNS=(
    # "logs/gadf_soil_nir_joint_resume_lowlr_base600/gadf_joint_resume_lowlr-ep700.pth.tar"
    "logs/gadf_soil_nir_joint_proj/gadf_joint_proj_20260413_172823/gadf_joint_proj-ep550.pth.tar"
    "logs/gadf_soil_nir_joint_transformer/gadf_joint_transformer_20260414_145223/gadf_joint_transformer-latest.pth.tar"
)

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Split:   $SPLIT  |  k_spt=$K_SPT  |  k_qry=$K_QRY"
echo "=========================================="

TOTAL_RUNS=0
SKIPPED=0

for CKPT in "${RUNS[@]}"; do

    if [ ! -f "${CKPT}" ]; then
        echo "WARN: Checkpoint not found: ${CKPT} — skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # El config params-joint.yaml se guarda en el mismo directorio que el checkpoint
    CKPT_DIR=$(dirname "${CKPT}")
    CONFIG="${CKPT_DIR}/params-joint.yaml"

    if [ ! -f "${CONFIG}" ]; then
        echo "WARN: Config not found: ${CONFIG} — skipping."
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    CKPT_NAME=$(basename "${CKPT_DIR}")
    SAVE="results/eval_joint_proto/${CKPT_NAME}_${SPLIT}_k${K_SPT}/"

    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    echo ""
    echo "[RUN ${TOTAL_RUNS}] ${CKPT_NAME}"
    echo "  ckpt   -> ${CKPT}"
    echo "  config -> ${CONFIG}"
    echo "  save   -> ${SAVE}"

    /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/bin/python main_eval_joint_protonet.py \
        --checkpoint "${CKPT}" \
        --config     "${CONFIG}" \
        --split      "${SPLIT}" \
        --data_path  "${DATA_PATH}" \
        --k_spt      "${K_SPT}" \
        --k_qry      "${K_QRY}" \
        --batch_size "${BATCH_SIZE}" \
        --save_path  "${SAVE}" \
        --device     cuda

done

echo ""
echo "=========================================="
echo "Eval ProtoNet finalizado."
echo "  Runs ejecutados : ${TOTAL_RUNS}"
echo "  Skipped         : ${SKIPPED}"
echo "  Resultados en   : results/eval_joint_proto/"
echo "=========================================="
