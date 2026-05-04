#!/bin/bash

#SBATCH --job-name=ft_gadf2d
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/finetune_gadf2d_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/finetune_gadf2d_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

# ===================================================================
# ijepa_spectra2d — Downstream fine-tuning con imágenes GADF 2D
#
# Pipeline:
#   1. Pretrained: full fine-tuning con 3 LRs
#   2. Pretrained: linear probing
#   3. No pretrain (baseline): fine-tuning con pesos aleatorios
#
# Usa GADF 2D pre-computados (.pt) desde data/Soil_NIR_AGG_mixed_gadf2d/
# (generados con scripts/precompute_gadf_downstream.py).
#
# Usage:
#   sbatch slurm_finetune_gadf2d.sh
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
SPLIT="test"

DOWNSTREAM_LRS=(1e-3 1e-4 1e-5)

# Checkpoint de pretrain I-JEPA 2D
PRETRAIN_CKPTS=(
    "logs/gadf_soil_nir/gadf_jepa_20260409_085855/gadf_jepa-latest.pth.tar"
)

# Modelo (debe coincidir con el config de pretraining)
MODEL_NAME="vit_base"
PATCH_SIZE=16
CROP_SIZE=224

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "=========================================="

TOTAL_RUNS=0

# ==============================================================
# 1. PRETRAINED: fine-tuning con cada checkpoint × cada LR
# ==============================================================
for CKPT in "${PRETRAIN_CKPTS[@]}"; do

    if [ ! -f "${CKPT}" ]; then
        echo "WARN: Checkpoint not found: ${CKPT}, skipping."
        continue
    fi

    CKPT_NAME=$(basename $(dirname "${CKPT}"))

    for ds_lr in "${DOWNSTREAM_LRS[@]}"; do
        SAVE="results/region_gadf2d/ft_${CKPT_NAME}_ftlr${ds_lr}/"
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
        echo ""
        echo "[RUN ${TOTAL_RUNS}] PRETRAINED fine-tuning: ckpt=${CKPT_NAME}, lr=${ds_lr}"
        echo "  -> ${SAVE}"

        python main_finetune_gadf2d.py \
            --data_path ${DATA_PATH} \
            --region_tasks \
            --split ${SPLIT} \
            --device cuda \
            --epoch 1000 \
            --batch_size 16 \
            --lr ${ds_lr} \
            --wd 1e-5 \
            --model_weight "${CKPT}" \
            --model_name ${MODEL_NAME} \
            --patch_size ${PATCH_SIZE} \
            --crop_size ${CROP_SIZE} \
            --patience 30 \
            --k_spt 25 \
            --k_qry 25 \
            --save_path "${SAVE}"
    done

    # ---- Linear probing (solo lr=1e-3, encoder congelado) ----
    SAVE_LP="results/region_gadf2d/linprobe_${CKPT_NAME}_lr1e-3/"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    echo ""
    echo "[RUN ${TOTAL_RUNS}] LINEAR PROBING: ckpt=${CKPT_NAME}"
    echo "  -> ${SAVE_LP}"

    python main_finetune_gadf2d.py \
        --data_path ${DATA_PATH} \
        --region_tasks \
        --split ${SPLIT} \
        --device cuda \
        --epoch 1000 \
        --batch_size 16 \
        --lr 1e-3 \
        --wd 1e-5 \
        --model_weight "${CKPT}" \
        --model_name ${MODEL_NAME} \
        --patch_size ${PATCH_SIZE} \
        --crop_size ${CROP_SIZE} \
        --linear_probing \
        --patience 30 \
        --k_spt 25 \
        --k_qry 25 \
        --save_path "${SAVE_LP}"

done

# ==============================================================
# 2. NO PRETRAIN (baseline): fine-tuning con pesos aleatorios
# ==============================================================
for ds_lr in "${DOWNSTREAM_LRS[@]}"; do
    SAVE_NP="results/region_gadf2d/nopret_ft_ftlr${ds_lr}/"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    echo ""
    echo "[RUN ${TOTAL_RUNS}] NO PRETRAIN fine-tuning: lr=${ds_lr}"
    echo "  -> ${SAVE_NP}"

    python main_finetune_gadf2d.py \
        --data_path ${DATA_PATH} \
        --region_tasks \
        --split ${SPLIT} \
        --device cuda \
        --epoch 1000 \
        --batch_size 16 \
        --lr ${ds_lr} \
        --wd 1e-5 \
        --model_weight '' \
        --model_name ${MODEL_NAME} \
        --patch_size ${PATCH_SIZE} \
        --crop_size ${CROP_SIZE} \
        --patience 30 \
        --k_spt 25 \
        --k_qry 25 \
        --save_path "${SAVE_NP}"
done

echo ""
echo "=========================================="
echo "Todos los experimentos de fine-tuning GADF 2D finalizados (${TOTAL_RUNS} runs)."
echo "=========================================="
