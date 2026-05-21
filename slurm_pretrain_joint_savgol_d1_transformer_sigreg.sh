#!/bin/bash
#SBATCH --job-name=joint_sg_d1_sigreg
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
# + Weak-SIGReg. alpha=1e-3, loss_cap=50, warmup=5.
# El clamp evita que picos de loss_sigreg (>50k observados en a1e-4)
# desestabilicen el entrenamiento; alpha 10x mayor para señal efectiva.
# ===================================================================

set -eo pipefail

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

RUN_ID="ep800_sigreg_a1e-3_cap50_wu5"
BASE_LOG="$(pwd)/logs/gadf_soil_nir_joint_savgol_d1_transformer_sigreg"
mkdir -p "${BASE_LOG}"

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Config:  configs/gadf_soil_nir_joint_savgol_d1_transformer_sigreg.yaml"
echo "Data:    data/gadf_224_v2_savgol_d1.h5"
echo "Tasks:   data/SoilDataset_NIR_gadf2d_savgol_d1"
echo "Mode:    joint + savgol d1 + transformer + Weak-SIGReg"
echo "SIGReg:  alpha=1e-3, loss_cap=50, sketch_dim=64, warmup=5, grad_clip=1.0"
echo "=========================================="

python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint_savgol_d1_transformer_sigreg.yaml \
    --devices cuda:0 \
    --run_id "${RUN_ID}" \
    --epochs 800 \
    --final_lr 5e-6 \
    --sigreg_alpha 0.001 \
    --sigreg_loss_cap 50.0 \
    --sigreg_warmup 5 \
    --sigreg_grad_clip_norm 1.0 \
    --wandb_name "joint_savgol_d1_transformer_sigreg_a1e-3_cap50_wu5_ep800"

echo "=========================================="
echo "Entrenamiento finalizado. Job $SLURM_JOB_ID done."
echo "=========================================="
