#!/bin/bash

#SBATCH --job-name=patch_probe_lr
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/patch_probe_lr_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/patch_probe_lr_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=e.ivangarzon98@go.ugr.es

# ===================================================================
# ijepa_spectra2d — LR sweep para per-patch linear probing
#
# Barre lr in {1e-4, 1e-3, 1e-2} sobre un subconjunto de tareas
# (--max_tasks 5, --n_repeats 1) para elegir el LR óptimo antes
# de lanzar el experimento completo.
#
# Usage:
#   sbatch slurm_per_patch_probe_lr_sweep.sh [split]
#   split: train | val | test  (default: val)
# ===================================================================

SPLIT=${1:-val}

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba
export LD_LIBRARY_PATH="/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib:$LD_LIBRARY_PATH"

cd /mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d
mkdir -p logs

echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "LR sweep — split=${SPLIT}, max_tasks=5, n_repeats=1"
echo "=========================================="

for LR in 1e-4 1e-3 1e-2; do
    echo ""
    echo "--- LR=${LR} ---"
    python scripts/per_patch_probe.py \
        --data_path data/SoilDataset_NIR_gadf2d \
        --region_tasks \
        --split ${SPLIT} \
        --device cuda \
        --model_weight logs/gadf_soil_nir_joint_resume_lowlr_base600/gadf_joint_resume_lowlr-ep700.pth.tar \
        --model_name vit_base \
        --patch_size 16 \
        --crop_size 224 \
        --gadf_norm_stats data/gadf_norm_stats_v2.json \
        --k_spt 250 \
        --k_qry 250 \
        --epoch 1000 \
        --batch_size 16 \
        --lr ${LR} \
        --wd 1e-5 \
        --patience 30 \
        --scheduler_patience 10 \
        --n_repeats 1 \
        --max_tasks 5 \
        --save_path results/lr_sweep_patch_probe/lr${LR}/
done

echo ""
echo "=========================================="
echo "Sweep finalizado. Comparativa de global avg pool R²:"
python3 - <<'EOF'
import json, os, glob

base = 'results/lr_sweep_patch_probe'
for lr_dir in sorted(glob.glob(f'{base}/lr*/summary.json')):
    with open(lr_dir) as f:
        s = json.load(f)
    lr = os.path.basename(os.path.dirname(lr_dir))
    print(f"  {lr}: global_avg_pool_R²={s['global_avg_pool_r2']:.4f}  "
          f"best_patch_R²={s['best_patch_r2']:.4f}")
EOF
echo "=========================================="
