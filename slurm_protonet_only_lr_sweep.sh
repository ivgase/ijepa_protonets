#!/bin/bash
#SBATCH --job-name=protonet_only_lr_sweep
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_protonet_only/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_protonet_only/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

# ===================================================================
# ijepa_spectra2d — LR grid search para ProtoNet-only fine-tuning
#
# Objetivo: encontrar el learning rate que maximiza R² en test tras
# unas pocas épocas de entrenamiento episódico ProtoNet puro sobre el
# target_encoder del checkpoint latest del joint resume
# gadf_joint_20260414_094939.
#
# Grid: 1e-6, 1e-5, 1e-4
#   - 3 valores conservadores para ampliar el sweep hacia un LR base clásico
#   - Encoder ya convergido → LR muy bajo esperado ganador
#   - Scheduler coherente por run: start_lr=lr y final_lr=lr/10
#   - Todos demás hiperparámetros fijos (epochs=100, warmup=2, etc.)
#
# Cada run escribe eval_test/results_per_task.csv con fila AVERAGE/MEAN.
# Al final se genera sweep_summary.csv ordenado por R² desc.
#
# Usage:
#   sbatch slurm_protonet_only_lr_sweep.sh
# ===================================================================

set -eo pipefail

# -- Entorno
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

# -- Directorio de logs base
mkdir -p logs/gadf_soil_nir_protonet_only

# -- Directorio del sweep (uno por job para no mezclar runs de jobs distintos)
SWEEP_DIR="logs/gadf_soil_nir_protonet_only/lr_sweep_${SLURM_JOB_ID}"
mkdir -p "$SWEEP_DIR"

MANIFEST="${SWEEP_DIR}/sweep_manifest.csv"
echo "lr,run_dir" > "$MANIFEST"

CHECKPOINT="/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint/gadf_joint_20260414_094939/gadf_joint-latest.pth.tar"

# -- Grid de learning rates
LR_VALUES=(1e-6 1e-5 1e-4)

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_protonet_only.yaml"
echo "GPU:      cuda:0"
echo "Ckpt:     ${CHECKPOINT}"
echo "Sweep:    LR over ${LR_VALUES[*]}"
echo "Output:   ${SWEEP_DIR}/"
echo "=========================================="

TOTAL_RUNS=0

# ==============================================================
# Sweep sobre learning rate
# ==============================================================
for LR in "${LR_VALUES[@]}"; do
    TAG="lr${LR}"
    TOTAL_RUNS=$((TOTAL_RUNS + 1))

    case "${LR}" in
        1e-6)
            FINAL_LR="1e-7"
            ;;
        1e-5)
            FINAL_LR="1e-6"
            ;;
        1e-4)
            FINAL_LR="1e-5"
            ;;
        *)
            echo "ERROR: no final_lr rule defined for lr=${LR}"
            exit 1
            ;;
    esac

    echo ""
    echo "================================================"
    echo "[RUN ${TOTAL_RUNS}/${#LR_VALUES[@]}] lr=${LR}  start_lr=${LR}  final_lr=${FINAL_LR}  tag=${TAG}"
    echo "================================================"

    python main_protonet_only.py \
        --fname configs/gadf_soil_nir_protonet_only.yaml \
        --devices cuda:0 \
        --checkpoint "${CHECKPOINT}" \
        --lr "${LR}" \
        --start_lr "${LR}" \
        --final_lr "${FINAL_LR}" \
        --epochs 100 \
        --folder "${SWEEP_DIR}" \
        --tag "${TAG}" \
        --wandb_name "proto_only_sweep_resume_latest_${TAG}"

    # main_protonet_only.py crea <folder>/<tag>_<timestamp>/
    RUN_DIR=$(ls -td "${SWEEP_DIR}/${TAG}_"*/ 2>/dev/null | head -1)
    if [ -n "$RUN_DIR" ]; then
        echo "lr,run_dir: ${LR},${RUN_DIR}"
        echo "${LR},${RUN_DIR}" >> "$MANIFEST"
        echo "  -> ${RUN_DIR}"
    else
        echo "WARN: no run dir found for tag=${TAG} in ${SWEEP_DIR}"
        echo "${LR}," >> "$MANIFEST"
    fi
done

echo ""
echo "=========================================="
echo "Todos los runs completados (${TOTAL_RUNS} LRs)."
# echo "Generando sweep_summary.csv ..."
echo "=========================================="

# # -- Agregación final: leer cada eval_test/results_per_task.csv y comparar R²
# python - <<PY
# import os
# import sys
# import pandas as pd

# manifest_path = "${MANIFEST}"
# sweep_dir     = "${SWEEP_DIR}"

# try:
#     manifest = pd.read_csv(manifest_path)
# except Exception as e:
#     print(f"ERROR: no se puede leer {manifest_path}: {e}")
#     sys.exit(1)

# rows = []
# for _, r in manifest.iterrows():
#     run_dir = str(r.get("run_dir", "")).strip().rstrip("/")
#     lr_val  = r.get("lr", "?")
#     if not run_dir:
#         print(f"WARN: run_dir vacío para lr={lr_val}")
#         continue
#     csv_path = os.path.join(run_dir, "eval_test", "results_per_task.csv")
#     if not os.path.exists(csv_path):
#         print(f"WARN: falta {csv_path}")
#         continue
#     try:
#         df = pd.read_csv(csv_path)
#     except Exception as e:
#         print(f"WARN: error leyendo {csv_path}: {e}")
#         continue
#     avg = df[(df["task"] == "AVERAGE") & (df["repeat"] == "MEAN")]
#     if avg.empty:
#         print(f"WARN: sin fila AVERAGE/MEAN en {csv_path}")
#         continue
#     avg = avg.iloc[0]
#     rows.append({
#         "lr":      lr_val,
#         "r2":      float(avg["r2"]),
#         "rmse":    float(avg["rmse"]),
#         "mae":     float(avg["mae"]),
#         "mse":     float(avg["mse"]),
#         "run_dir": run_dir,
#     })

# if not rows:
#     print("ERROR: ningún run produjo resultados válidos.")
#     sys.exit(1)

# results = pd.DataFrame(rows).sort_values("r2", ascending=False)
# out_path = os.path.join(sweep_dir, "sweep_summary.csv")
# results.to_csv(out_path, index=False)

# print()
# print("=" * 60)
# print("LR SWEEP SUMMARY (sorted by R² desc)")
# print("=" * 60)
# print(results[["lr", "r2", "rmse", "mae", "mse"]].to_string(index=False))
# best = results.iloc[0]
# print()
# print(f"Best LR:   {best['lr']}")
# print(f"Best R²:   {best['r2']:.6f}")
# print(f"Best RMSE: {best['rmse']:.6f}")
# print(f"Best run:  {best['run_dir']}")
# print("=" * 60)
# print(f"Resultados completos: {out_path}")
# PY

# echo ""
# echo "=========================================="
# echo "Sweep LR finalizado. Ver ${SWEEP_DIR}/sweep_summary.csv"
# echo "=========================================="
