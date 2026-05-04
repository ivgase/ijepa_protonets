#!/bin/bash

#SBATCH --job-name=joint_proj_proto
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/joint_proj_proto_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/joint_proj_proto_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

# ===================================================================
# ijepa_spectra2d — ProtoNet con ProjectionNet learnable
#                   sobre embeddings del checkpoint joint
#
# Pipeline:
#   1. Precomputa embeddings si no existen (los genera slurm_joint_zeroshot)
#   2. Entrena ProjectionNet (768->256->128) con meta-learning episodico
#      sobre los embeddings congelados del target_encoder joint
#
# NOTA: lanzar slurm_joint_zeroshot_protonet.sh PRIMERO para que los
# embeddings esten disponibles. Si no existen, este script los computa.
#
# Usage:
#   sbatch slurm_joint_proj_protonet.sh
# ===================================================================

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d
mkdir -p logs

# ------ Config ------
CHECKPOINT="logs/gadf_soil_nir_joint/gadf_joint_20260414_094939/gadf_joint-latest.pth.tar"
DATA_PATH="data/SoilDataset_NIR_gadf2d"
NORM_STATS="data/gadf_norm_stats_v2.json"
MODEL_NAME="vit_base"
EPISODES=5000
REPEATS=3
K_SPT=25
K_QRY=25
LR=0.001
DIST_TEMP=0.5
HIDDEN_DIM=256
PROJ_DIM=128
OUTPUT="results/joint_proj_protonet/"

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Checkpoint: $CHECKPOINT"
echo "epochs=${EPISODES} | repeats=${REPEATS} | k_spt=${K_SPT}"
echo "=========================================="

# ==============================================================
# Step 1: Precompute embeddings (skip si ya existen)
# ==============================================================
FIRST_TASK=$(ls -d ${DATA_PATH}/Soil_*/ | head -1)
EMB_FILE="${FIRST_TASK}emb_supp.pt"

if [ ! -f "${EMB_FILE}" ]; then
    echo ""
    echo "[STEP 1] Embeddings no encontrados, precomputando..."
    python scripts/precompute_embeddings.py \
        --checkpoint  "${CHECKPOINT}" \
        --data_path   "${DATA_PATH}" \
        --norm_stats  "${NORM_STATS}" \
        --model_name  "${MODEL_NAME}" \
        --batch_size  64 \
        --emb_mode    mean \
        --device      cuda \
        --overwrite
else
    echo "[STEP 1] Embeddings ya existen, saltando precomputo."
fi

# ==============================================================
# Step 2: Entrenar ProjectionNet con ProtoNet episodico
# ==============================================================
echo ""
echo "[STEP 2] Entrenando ProjectionNet (768->${HIDDEN_DIM}->${PROJ_DIM})..."
python main_protonet_gadf2d.py \
    --data_path    "${DATA_PATH}" \
    --output       "${OUTPUT}" \
    --episodes     "${EPISODES}" \
    --repeats      "${REPEATS}" \
    --k_spt        "${K_SPT}" \
    --k_qry        "${K_QRY}" \
    --lr           "${LR}" \
    --dist_temp    "${DIST_TEMP}" \
    --embed_dim    768 \
    --hidden_dim   "${HIDDEN_DIM}" \
    --proj_dim     "${PROJ_DIM}" \
    --val_freq     50 \
    --arch         mlp \
    --emb_mode     mean \
    --device       cuda \
    --wandb_project protonet-gadf2d \
    --wandb_name    "joint_noproj_lp0.025_proj768-${HIDDEN_DIM}-${PROJ_DIM}_ep${EPISODES}"

echo ""
echo "=========================================="
echo "ProjectionNet training finalizado."
echo "  Resultados en: ${OUTPUT}"
echo "=========================================="
