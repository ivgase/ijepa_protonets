#!/bin/bash
#SBATCH --job-name=protonet_only
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_protonet_only/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_protonet_only/slurm_%j.err
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es
#SBATCH --mail-type=END,FAIL

set -eo pipefail

# -- Entorno
export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d

# -- Directorio de logs
mkdir -p logs/gadf_soil_nir_protonet_only

# -- Info del job
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_protonet_only.yaml"
echo "GPU:      cuda:0"
echo "Objetivo: Fine-tuning ProtoNet-only sobre target_encoder del joint ckpt"
echo "Ckpt:     logs/gadf_soil_nir_joint/gadf_joint_20260414_094939/gadf_joint-latest.pth.tar"
echo "Epocas:   20  |  LR: 1e-4  |  loss: smooth_l1"
echo "=========================================="
echo ""
echo "NOTA: Al arrancar deben aparecer en el log:"
echo "  - 'Loaded target_encoder from epoch <N>'"
echo "  - 'ProtoNet sampler: <M> train tasks'"
echo "  - Primera epoch: 'Epoch 1/20'"
echo "=========================================="

python main_protonet_only.py \
    --fname configs/gadf_soil_nir_protonet_only.yaml \
    --devices cuda:0
