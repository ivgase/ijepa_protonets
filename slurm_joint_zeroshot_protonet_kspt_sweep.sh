#!/bin/bash

#SBATCH --job-name=joint_zero_kspt
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/joint_zero_kspt_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/joint_zero_kspt_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

# ===================================================================
# ijepa_spectra2d — Sweep de k_spt para ProtoNet zero-shot sobre
# checkpoint joint sin proyeccion (IdentityNet)
#
# Pipeline:
#   1. Precomputa embeddings del target_encoder una sola vez
#   2. Evalua ProtoNet zero-shot para varios tamaños de soporte
#
# Nota:
#   - El checkpoint objetivo no usa proto_projection, asi que reutilizamos
#     el flujo "no_projection" basado en embeddings precomputados.
#   - En val/test la evaluacion usa SIEMPRE el query completo de cada tarea;
#     k_qry se mantiene fijo solo por compatibilidad con el script.
#
# Usage:
#   sbatch slurm_joint_zeroshot_protonet_kspt_sweep.sh
# ===================================================================

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d
mkdir -p logs

# ------ Config ------
CHECKPOINT="/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_resume_lowlr_base600/gadf_joint_resume_lowlr-ep700.pth.tar"
DATA_PATH="data/SoilDataset_NIR_gadf2d"
NORM_STATS="data/gadf_norm_stats_v2.json"
MODEL_NAME="vit_base"
K_QRY=25
K_SPT_VALUES=(5 10 25 50 100 250 500)
OUTPUT_ROOT="results/joint_noproj_zeroshot_kspt_ep700"
EMBEDDINGS_ROOT="artifacts/embeddings/gadf_joint_resume_lowlr_ep700_mean"

echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Checkpoint:  $CHECKPOINT"
echo "k_spt sweep: ${K_SPT_VALUES[*]}"
echo "k_qry fixed: $K_QRY"
echo "embeddings:  $EMBEDDINGS_ROOT"
echo "=========================================="

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

# ==============================================================
# Step 1: Precompute embeddings
# Los embeddings se guardan fuera del dataset para no sobreescribir
# emb_supp.pt / emb_query.pt al cambiar de checkpoint.
# ==============================================================
echo ""
echo "[STEP 1] Precomputing embeddings from joint target_encoder..."
python scripts/precompute_embeddings.py \
    --checkpoint  "${CHECKPOINT}" \
    --data_path   "${DATA_PATH}" \
    --output_root "${EMBEDDINGS_ROOT}" \
    --norm_stats  "${NORM_STATS}" \
    --model_name  "${MODEL_NAME}" \
    --batch_size  64 \
    --emb_mode    mean \
    --device      cuda \
    --overwrite

# ==============================================================
# Step 2: Zero-shot ProtoNet eval sweep over k_spt
# --episodes 1 y --no_projection mantienen el experimento en modo
# identidad; no hay parametros entrenables.
# ==============================================================
echo ""
echo "[STEP 2] ProtoNet zero-shot sweep over k_spt..."

TOTAL_RUNS=0

for K_SPT in "${K_SPT_VALUES[@]}"; do
    OUTPUT="${OUTPUT_ROOT}/k${K_SPT}/"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    echo ""
    echo "[RUN ${TOTAL_RUNS}] k_spt=${K_SPT}, k_qry=${K_QRY}"
    echo "  -> ${OUTPUT}"

    python main_protonet_gadf2d.py \
        --data_path    "${DATA_PATH}" \
        --embeddings_root "${EMBEDDINGS_ROOT}" \
        --output       "${OUTPUT}" \
        --no_projection \
        --episodes     1 \
        --repeats      1 \
        --k_spt        "${K_SPT}" \
        --k_qry        "${K_QRY}" \
        --dist_temp    0.5 \
        --device       cuda
done

echo ""
echo "=========================================="
echo "Sweep ProtoNet zero-shot finalizado."
echo "  Runs ejecutados : ${TOTAL_RUNS}"
echo "  Resultados en   : ${OUTPUT_ROOT}/k*/"
echo "=========================================="
