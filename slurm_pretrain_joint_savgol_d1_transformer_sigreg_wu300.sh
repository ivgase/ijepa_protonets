#!/bin/bash
#SBATCH --job-name=joint_sg_d1_sigreg_a1e-4
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_savgol_d1_transformer_sigreg/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint_savgol_d1_transformer_sigreg/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

# ===================================================================
# Joint I-JEPA + ProtoNet + Savitzky-Golay deriv=1 + TransformerAggregator
# + Weak-SIGReg. alpha=1e-4, loss_cap=10, warmup=50.
#
# Motivación: análisis de balance de losses sobre el run 133326 (a1e-3,
# cap50, wu5) mostró que sigreg ocupaba el 70-100% del gradiente JEPA
# durante ep5-300 (25x el loss_jepa). El run wu300 mejora la fase inicial
# pero con alpha=1e-3 sigreg sigue siendo 5x loss_jepa en el estado estable.
#
# Calibración correcta:
#   alpha * loss_sigreg_natural(~7.8) ≈ 0.3-0.5 × loss_jepa_convergido(~0.002)
#   → alpha ≈ 1e-4
# Con alpha=1e-4 y cap=10:
#   - durante spikes: 1e-4 × 10 = 0.001 << loss_jepa inicial (0.015) ✓
#   - fase estable:   1e-4 × 7.8 = 0.00078 ≈ 40% loss_jepa (0.002)  ✓
#   - proto (sep.):   0.025 × 0.12 = 0.003 → comparable ✓
# warmup=50 es suficiente porque la contribución máxima ya es pequeña.
# ===================================================================

set -eo pipefail

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

RUN_ID="ep1200_sigreg_a1e-4_cap10_wu50"
BASE_LOG="$(pwd)/logs/gadf_soil_nir_joint_savgol_d1_transformer_sigreg"
mkdir -p "${BASE_LOG}"

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  configs/gadf_soil_nir_joint_savgol_d1_transformer_sigreg.yaml"
echo "Data:    data/gadf_224_v2_savgol_d1.h5"
echo "Tasks:   data/SoilDataset_NIR_gadf2d_savgol_d1"
echo "Mode:    joint + savgol d1 + transformer + Weak-SIGReg"
echo "SIGReg:  alpha=1e-4, loss_cap=10, sketch_dim=64, warmup=50, grad_clip=1.0"
echo "=========================================="

python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint_savgol_d1_transformer_sigreg.yaml \
    --devices cuda:0 \
    --run_id "${RUN_ID}" \
    --epochs 1200 \
    --final_lr 5e-6 \
    --sigreg_alpha 0.0001 \
    --sigreg_loss_cap 10.0 \
    --sigreg_warmup 50 \
    --sigreg_grad_clip_norm 1.0 \
    --wandb_name "joint_savgol_d1_transformer_sigreg_a1e-4_cap10_wu50_ep1200"

echo "=========================================="
echo "Entrenamiento finalizado. Job $SLURM_JOB_ID done."
echo "=========================================="
