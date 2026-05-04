#!/bin/bash

#SBATCH --job-name=lora_gadf2d
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/lora_gadf2d_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/lora_gadf2d_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

# ===================================================================
# ijepa_spectra2d — LoRA fine-tuning con imágenes GADF 2D
#
# Pipeline:
#   1. Pretrained: LoRA fine-tuning con sweep de ranks y LRs
#   2. No pretrain (baseline): LoRA con pesos aleatorios
#
# Usa GADF 2D pre-computados (.pt) desde data/Soil_NIR_AGG_mixed_gadf2d/
#
# Usage:
#   sbatch slurm_lora_gadf2d.sh
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

LORA_LRS=(1e-3 1e-4 1e-5)
LORA_RANKS=(8)
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Checkpoint de pretrain I-JEPA 2D (ruta fija del run original)
PRETRAIN_CKPTS=(
    "logs/gadf_soil_nir/gadf_jepa-latest.pth.tar"
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
# 1. PRETRAINED: LoRA fine-tuning con cada checkpoint × rank × LR
# ==============================================================
for CKPT in "${PRETRAIN_CKPTS[@]}"; do

    if [ ! -f "${CKPT}" ]; then
        echo "WARN: Checkpoint not found: ${CKPT}, skipping."
        continue
    fi

    CKPT_NAME=$(basename $(dirname "${CKPT}"))

    for lora_r in "${LORA_RANKS[@]}"; do
        for ds_lr in "${LORA_LRS[@]}"; do
            SAVE="results/region_gadf2d/lora_${CKPT_NAME}_r${lora_r}_lr${ds_lr}/"
            TOTAL_RUNS=$((TOTAL_RUNS + 1))
            echo ""
            echo "[RUN ${TOTAL_RUNS}] LoRA fine-tuning: ckpt=${CKPT_NAME}, r=${lora_r}, lr=${ds_lr}"
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
                --use_lora \
                --lora_r ${lora_r} \
                --lora_alpha ${LORA_ALPHA} \
                --lora_dropout ${LORA_DROPOUT} \
                --patience 30 \
                --k_spt 25 \
                --k_qry 25 \
                --save_path "${SAVE}"
        done
    done

done

# # ==============================================================
# # 2. NO PRETRAIN (baseline): LoRA con pesos aleatorios
# # ==============================================================
# for lora_r in "${LORA_RANKS[@]}"; do
#     for ds_lr in "${LORA_LRS[@]}"; do
#         SAVE_NP="results/region_gadf2d/nopret_lora_r${lora_r}_lr${ds_lr}/"
#         TOTAL_RUNS=$((TOTAL_RUNS + 1))
#         echo ""
#         echo "[RUN ${TOTAL_RUNS}] NO PRETRAIN LoRA: r=${lora_r}, lr=${ds_lr}"
#         echo "  -> ${SAVE_NP}"

#         python main_finetune_gadf2d.py \
#             --data_path ${DATA_PATH} \
#             --region_tasks \
#             --split ${SPLIT} \
#             --device cuda \
#             --epoch 1000 \
#             --batch_size 16 \
#             --lr ${ds_lr} \
#             --wd 1e-5 \
#             --model_weight '' \
#             --model_name ${MODEL_NAME} \
#             --patch_size ${PATCH_SIZE} \
#             --crop_size ${CROP_SIZE} \
#             --use_lora \
#             --lora_r ${lora_r} \
#             --lora_alpha ${LORA_ALPHA} \
#             --lora_dropout ${LORA_DROPOUT} \
#             --patience 30 \
#             --k_spt 25 \
#             --k_qry 25 \
#             --save_path "${SAVE_NP}"
#     done
# done

echo ""
echo "=========================================="
echo "Todos los experimentos LoRA GADF 2D finalizados (${TOTAL_RUNS} runs)."
echo "=========================================="
