#!/bin/bash

#SBATCH --job-name=eval_joint_savgol_d1
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/eval_joint_proto_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/eval_joint_proto_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

# ===================================================================
# Eval ProtoNet zero-shot — joint + Savitzky-Golay derivada 1 (ep750)
# Mejor probe val R²=0.4812 (ep375) / 0.4496 (ep700). Dataset: savgol_d1.
# use_projection=false → flujo de dos pasos:
#   1. precompute_embeddings.py  → artifacts/embeddings/
#   2. main_protonet_gadf2d.py --no_projection
# ===================================================================

set +u
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
set -euo pipefail
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

CKPT="logs/gadf_soil_nir_joint_savgol_d1/gadf_joint_savgol_d1_ep800/gadf_joint_savgol_d1-ep750.pth.tar"
DATA_PATH="data/SoilDataset_NIR_gadf2d_savgol_d1"
NORM_STATS="data/gadf_norm_stats_v2_savgol_d1.json"
EMB_ROOT="artifacts/embeddings/gadf_joint_savgol_d1_ep750_mean"
K_SPT=25
K_QRY=25
BATCH_SIZE=64
OUTPUT="results/eval_joint_proto/gadf_joint_savgol_d1_ep750_test_k${K_SPT}/"

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Ckpt:    $CKPT"
echo "k_spt=$K_SPT  |  k_qry=$K_QRY"
echo "Output:  $OUTPUT"
echo "=========================================="

# ==============================================================
# Step 1: Precompute embeddings
# ==============================================================
echo ""
echo "[STEP 1] Precomputing embeddings from joint savgol_d1 target_encoder..."
/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/bin/python scripts/precompute_embeddings.py \
    --checkpoint  "${CKPT}" \
    --data_path   "${DATA_PATH}" \
    --norm_stats  "${NORM_STATS}" \
    --model_name  vit_base \
    --batch_size  "${BATCH_SIZE}" \
    --emb_mode    mean \
    --output_root "${EMB_ROOT}" \
    --device      cuda \
    --overwrite

# ==============================================================
# Step 2: Zero-shot ProtoNet eval
# ==============================================================
echo ""
echo "[STEP 2] Zero-shot ProtoNet eval (no_projection)..."
/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/bin/python main_protonet_gadf2d.py \
    --data_path       "${DATA_PATH}" \
    --embeddings_root "${EMB_ROOT}" \
    --output          "${OUTPUT}" \
    --no_projection \
    --episodes        1 \
    --repeats         1 \
    --k_spt           "${K_SPT}" \
    --k_qry           "${K_QRY}" \
    --dist_temp       0.5 \
    --device          cuda

echo ""
echo "=========================================="
echo "Eval finalizado. Resultados en: ${OUTPUT}"
echo "=========================================="
