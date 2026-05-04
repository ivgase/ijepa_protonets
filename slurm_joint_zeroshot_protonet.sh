#!/bin/bash

#SBATCH --job-name=joint_zeroshot
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/joint_zeroshot_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/joint_zeroshot_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

# ===================================================================
# ijepa_spectra2d — ProtoNet zero-shot eval sobre checkpoint joint
#                   sin proyeccion (IdentityNet)
#
# Pipeline:
#   1. Precomputa embeddings del target_encoder del checkpoint joint
#      (--overwrite para garantizar que corresponden a este checkpoint)
#   2. Evalua ProtoNet zero-shot: val + test, sin ningun entrenamiento
#
# NOTA: lanzar este script ANTES que slurm_joint_proj_protonet.sh,
# porque el paso 1 genera los emb_supp.pt que el otro reutiliza.
#
# Usage:
#   sbatch slurm_joint_zeroshot_protonet.sh
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
K_SPT=25
K_QRY=25
OUTPUT="results/joint_noproj_zeroshot/"

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

# ==============================================================
# Step 1: Precompute embeddings
# --overwrite garantiza que los embeddings son de este checkpoint
# ==============================================================
echo ""
echo "[STEP 1] Precomputing embeddings from joint target_encoder..."
python scripts/precompute_embeddings.py \
    --checkpoint  "${CHECKPOINT}" \
    --data_path   "${DATA_PATH}" \
    --norm_stats  "${NORM_STATS}" \
    --model_name  "${MODEL_NAME}" \
    --batch_size  64 \
    --emb_mode    mean \
    --device      cuda \
    --overwrite

# ==============================================================
# Step 2: Zero-shot ProtoNet eval (sin proyeccion, sin training)
# --episodes 1 es suficiente: val se evalua en episode 0,
# test se evalua al salir del bucle. No hay nada que aprender.
# ==============================================================
echo ""
echo "[STEP 2] Zero-shot ProtoNet eval (no_projection)..."
python main_protonet_gadf2d.py \
    --data_path    "${DATA_PATH}" \
    --output       "${OUTPUT}" \
    --no_projection \
    --episodes     1 \
    --repeats      1 \
    --k_spt        "${K_SPT}" \
    --k_qry        "${K_QRY}" \
    --dist_temp    0.5 \
    --device       cuda

echo ""
echo "=========================================="
echo "Zero-shot eval finalizado."
echo "  Resultados en: ${OUTPUT}"
echo "=========================================="
