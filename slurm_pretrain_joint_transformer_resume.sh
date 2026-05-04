#!/bin/bash
#SBATCH --job-name=joint_transformer_resume
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_transformer_resume/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_transformer_resume/slurm_%j.err
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
# Checkpoint base: job 129190, épocas 0→400 (sin positional embedding aprendible)
BASE_CKPT="/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_transformer/gadf_joint_transformer_20260421_061332/gadf_joint_transformer-latest.pth.tar"

RUN_ID="ep400base"
TAG="gadf_joint_transformer_resume"
BASE_LOG="$(pwd)/logs/gadf_soil_nir_joint_transformer_resume"
RUN_DIR="${BASE_LOG}/${TAG}_${RUN_ID}"
CKPT="${RUN_DIR}/${TAG}-latest.pth.tar"

mkdir -p "${BASE_LOG}"

echo "=========================================="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "Config:      configs/gadf_soil_nir_joint_transformer_resume.yaml"
echo "GPU:         cuda:0"
echo "Base ckpt:   ${BASE_CKPT}"
echo "Run dir:     ${RUN_DIR}"
echo "Flujo:       ep400→600 (coseno) | ep600→800 (lowlr)"
echo "=========================================="

# ── Etapa 1: resume 400 → 600 épocas (coseno rescalado) ──────────────────────
echo ""
echo ">>> ETAPA 1: resume 400 → 600 épocas"
python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint_transformer_resume.yaml \
    --devices cuda:0 \
    --run_id "${RUN_ID}" \
    --read_checkpoint "${BASE_CKPT}" \
    --epochs 600 \
    --wandb_name "joint_transformer_resume_ep600"

echo ">>> ETAPA 1 completada. Checkpoint: ${CKPT}"

# ── Etapa 2: resume 600 → 800 épocas (lowlr) ─────────────────────────────────
echo ""
echo ">>> ETAPA 2: resume 600 → 800 épocas (lowlr)"
python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint_transformer_resume.yaml \
    --devices cuda:0 \
    --run_id "${RUN_ID}" \
    --read_checkpoint "${CKPT}" \
    --epochs 800 \
    --lr 5e-4 \
    --start_lr 5e-5 \
    --final_lr 5e-7 \
    --warmup 10 \
    --wandb_name "joint_transformer_resume_ep800_lowlr"

echo ""
echo "=========================================="
echo "Resume transformer finalizado. Job $SLURM_JOB_ID done."
echo "=========================================="
