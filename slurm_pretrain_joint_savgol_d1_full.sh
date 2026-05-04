#!/bin/bash
#SBATCH --job-name=pretrain_joint_savgol_d1
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_savgol_d1/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_savgol_d1/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# ── Entorno ──────────────────────────────────────────────────────────────────
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

# ── Rutas ─────────────────────────────────────────────────────────────────────
RUN_ID="ep800"
TAG="gadf_joint_savgol_d1"
BASE_LOG="$(pwd)/logs/gadf_soil_nir_joint_savgol_d1"
RUN_DIR="${BASE_LOG}/${TAG}_${RUN_ID}"
CKPT="${RUN_DIR}/${TAG}-latest.pth.tar"

mkdir -p "${BASE_LOG}"

echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_joint_savgol_d1.yaml"
echo "Run dir:  ${RUN_DIR}"
echo "GPU:      cuda:0"
echo "Flujo:    ep0→400 | resume ep600 | resume lowlr ep800"
echo "=========================================="

# ── Etapa 1: entrenamiento base 0 → 400 épocas ───────────────────────────────
echo ""
echo ">>> ETAPA 1: 0 → 400 épocas"
python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint_savgol_d1.yaml \
    --devices cuda:0 \
    --run_id "${RUN_ID}" \
    --epochs 400 \
    --wandb_name "joint_savgol_d1_ep400"

echo ">>> ETAPA 1 completada. Checkpoint: ${CKPT}"

# ── Etapa 2: resume 400 → 600 épocas (coseno rescalado) ──────────────────────
echo ""
echo ">>> ETAPA 2: resume 400 → 600 épocas"
python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint_savgol_d1.yaml \
    --devices cuda:0 \
    --read_checkpoint "${CKPT}" \
    --epochs 600 \
    --wandb_name "joint_savgol_d1_ep600"

echo ">>> ETAPA 2 completada."

# ── Etapa 3: resume 600 → 800 épocas (lowlr) ─────────────────────────────────
echo ""
echo ">>> ETAPA 3: resume 600 → 800 épocas (lowlr)"
python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint_savgol_d1.yaml \
    --devices cuda:0 \
    --read_checkpoint "${CKPT}" \
    --epochs 800 \
    --lr 5e-4 \
    --start_lr 5e-5 \
    --final_lr 5e-7 \
    --warmup 10 \
    --wandb_name "joint_savgol_d1_ep800_lowlr"

echo ">>> ETAPA 3 completada. Entrenamiento savgol_d1 finalizado."
