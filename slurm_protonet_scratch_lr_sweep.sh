#!/bin/bash
#SBATCH --job-name=proto_scratch_sweep
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_protonet_scratch/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_protonet_scratch/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

# ===================================================================
# ijepa_spectra2d — LR sweep ProtoNet from scratch (ViT-base, GADF)
#
# Objetivo: encontrar el mejor LR para entrenar ProtoNets con ViT-base
# y entradas GADF 2D desde pesos aleatorios, sin ningún pre-training.
#
# Proporciona el baseline "mismo encoder, sin I-JEPA" para la
# comparativa justa con:
#   - fewshot_nir_orig (ResNet1D, espectros raw)
#   - joint ijepa_spectra2d (ViT-base, con I-JEPA pre-training)
#
# Grid: 1e-2, 1e-3, 1e-4
#   - Cubre desde LR agresivo (comparable a 5e-3 en fewshot_nir_orig)
#     hasta el rango típico de ViT con AdamW + cosine
#   - start_lr = lr / 100  (warmup desde casi 0)
#   - final_lr = lr / 100  (misma regla que slurm_protonet_only_lr_sweep.sh)
#   - 1000 epochs ≈ 15k pasos con meta_batch=8
#
# Cada run escribe eval_test/results_per_task.csv (fila AVERAGE/MEAN).
#
# Usage:
#   ./launch.sh slurm_protonet_scratch_lr_sweep.sh "LR sweep protonet from scratch ViT-GADF"
# ===================================================================

set -eo pipefail

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

mkdir -p logs/gadf_soil_nir_protonet_scratch

SWEEP_DIR="logs/gadf_soil_nir_protonet_scratch/lr_sweep_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR"

MANIFEST="${SWEEP_DIR}/sweep_manifest.csv"
echo "lr,start_lr,final_lr,run_dir" > "$MANIFEST"

LR_VALUES=(1e-2 1e-3 1e-4)

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_protonet_scratch.yaml"
echo "GPU:      cuda:0"
echo "Arquitectura: vit_base, patch=16, GADF 224x224, random init"
echo "Sweep:    LR over ${LR_VALUES[*]}"
echo "Epochs:   1000 por run"
echo "Output:   ${SWEEP_DIR}/"
echo "=========================================="

TOTAL_RUNS=0

for LR in "${LR_VALUES[@]}"; do
    TAG="lr${LR}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    # start_lr y final_lr = lr / 100
    case "${LR}" in
        1e-2)
            START_LR="1e-4"
            FINAL_LR="1e-4"
            ;;
        1e-3)
            START_LR="1e-5"
            FINAL_LR="1e-5"
            ;;
        1e-4)
            START_LR="1e-6"
            FINAL_LR="1e-6"
            ;;
        *)
            echo "ERROR: no final_lr rule defined for lr=${LR}"
            exit 1
            ;;
    esac

    echo ""
    echo "================================================"
    echo "[RUN ${TOTAL_RUNS}/${#LR_VALUES[@]}] lr=${LR}  start_lr=${START_LR}  final_lr=${FINAL_LR}  tag=${TAG}"
    echo "================================================"

    python main_protonet_only.py \
        --fname configs/gadf_soil_nir_protonet_scratch.yaml \
        --devices cuda:0 \
        --lr "${LR}" \
        --start_lr "${START_LR}" \
        --final_lr "${FINAL_LR}" \
        --epochs 1000 \
        --folder "${SWEEP_DIR}" \
        --tag "${TAG}" \
        --wandb_name "proto_scratch_sweep_${TAG}_${SLURM_JOB_ID}"

    RUN_DIR=$(ls -td "${SWEEP_DIR}/${TAG}_"*/ 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ]; then
        echo "${LR},${START_LR},${FINAL_LR},${RUN_DIR}" >> "$MANIFEST"
        echo "  -> ${RUN_DIR}"
    else
        echo "WARN: no run dir found for tag=${TAG} in ${SWEEP_DIR}"
        echo "${LR},${START_LR},${FINAL_LR}," >> "$MANIFEST"
    fi
done

echo ""
echo "=========================================="
echo "Todos los runs completados (${TOTAL_RUNS} LRs)."
echo "Manifest: ${MANIFEST}"
echo "Para comparar R² entre runs:"
echo "  grep 'AVERAGE' ${SWEEP_DIR}/lr*/eval_test/results_per_task.csv"
echo "=========================================="
