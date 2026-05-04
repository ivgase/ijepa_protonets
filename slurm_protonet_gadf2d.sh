#!/bin/bash

#SBATCH --job-name=proto_gadf2d
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/protonet_gadf2d_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/protonet_gadf2d_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

# ===================================================================
# ijepa_spectra2d — ProtoNet few-shot regression on I-JEPA embeddings
#
# Pipeline:
#   1. Precompute embeddings (skip if already done)
#   2. Train ProtoNet with different hyperparameters
#
# Usage:
#   sbatch slurm_protonet_gadf2d.sh
# ===================================================================

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d
mkdir -p logs

# ------ Config ------
DATA_PATH="data/SoilDataset_NIR_gadf2d"
CHECKPOINT="logs/gadf_soil_nir/gadf_jepa-latest.pth.tar"
MODEL_NAME="vit_base"
EPISODES=10000
REPEATS=1
K_SPT=25
K_QRY=25
LR=0.001
DIST_TEMP=0.5

# Architecture: mlp | resnet1d
ARCH=${ARCH:-mlp}
# Embedding mode: mean | patches
EMB_MODE=${EMB_MODE:-mean}

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "=========================================="

# ==============================================================
# Step 1: Precompute embeddings (skip if already done)
# ==============================================================
FIRST_TASK=$(ls -d ${DATA_PATH}/Soil_*/ | head -1)
if [ "${EMB_MODE}" = "patches" ]; then
    EMB_FILE="${FIRST_TASK}emb_supp_patches.pt"
else
    EMB_FILE="${FIRST_TASK}emb_supp.pt"
fi

if [ ! -f "${EMB_FILE}" ]; then
    echo ""
    echo "[STEP 1] Precomputing I-JEPA embeddings (mode=${EMB_MODE})..."
    python scripts/precompute_embeddings.py \
        --data_path ${DATA_PATH} \
        --checkpoint ${CHECKPOINT} \
        --model_name ${MODEL_NAME} \
        --batch_size 64 \
        --emb_mode ${EMB_MODE} \
        --device cuda
else
    echo "[STEP 1] Embeddings already exist (mode=${EMB_MODE}), skipping."
fi

# ==============================================================
# Step 2: Train ProtoNet
# ==============================================================
SAVE="results/protonet_gadf2d/${ARCH}_${EMB_MODE}/"
echo ""
echo "[STEP 2] Training ProtoNet | arch=${ARCH} | emb_mode=${EMB_MODE}"
echo "  -> ${SAVE}"

python main_protonet_gadf2d.py \
    --data_path ${DATA_PATH} \
    --output ${SAVE} \
    --episodes ${EPISODES} \
    --lr ${LR} \
    --k_spt ${K_SPT} \
    --k_qry ${K_QRY} \
    --dist_temp ${DIST_TEMP} \
    --repeats ${REPEATS} \
    --arch ${ARCH} \
    --emb_mode ${EMB_MODE} \
    --device cuda \
    --wandb_project protonet-gadf2d \
    --wandb_name "${ARCH}_${EMB_MODE}"

echo ""
echo "=========================================="
echo "ProtoNet experiment finished."
echo "=========================================="
