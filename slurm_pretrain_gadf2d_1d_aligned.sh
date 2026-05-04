#!/bin/bash
#SBATCH --job-name=pretrain_gadf2d_1d_aligned
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_1d_aligned/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_1d_aligned/slurm_%j.err
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

# ── Directorio de logs ────────────────────────────────────────────────────────
mkdir -p logs/gadf_soil_nir_1d_aligned

# ── Info del job ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_1d_aligned.yaml"
echo "Model:    vit_1d_aligned (embed_dim=256, depth=8, heads=8)"
echo "Purpose:  Ablation — match 1D SpectraI-JEPA architecture"
echo "GPU:      cuda:0"
echo "=========================================="

# ── Pretraining ───────────────────────────────────────────────────────────────
python main.py \
    --fname configs/gadf_soil_nir_1d_aligned.yaml \
    --devices cuda:0

echo "=========================================="
echo "Pretraining finished. Job $SLURM_JOB_ID done."
echo "=========================================="
