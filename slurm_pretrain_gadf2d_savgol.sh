#!/bin/bash
#SBATCH --job-name=pretrain_savgol
#SBATCH --partition=dgx2,dgx
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=50G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_savgol/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_savgol/slurm_%j.err
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
mkdir -p logs/gadf_soil_nir_savgol

# ── Info del job ──────────────────────────────────────────────────────────────
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_savgol.yaml"
echo "GPU:      cuda:0"
echo "=========================================="

# ── Pretraining ───────────────────────────────────────────────────────────────
RUN_DIR="logs/gadf_soil_nir_savgol/gadf_jepa_savgol_${SLURM_JOB_ID}"
BEST_CKPT="${RUN_DIR}/gadf_jepa_savgol-best_probe.pth.tar"

python main.py \
    --fname configs/gadf_soil_nir_savgol.yaml \
    --devices cuda:0 \
    --epochs 200 \
    --run_id "${SLURM_JOB_ID}" \
    --wandb_name "ijepa_savgol_ep200"

echo "=========================================="
echo "Pretraining finished."
echo "=========================================="

# ── Auto fine-tuning con el mejor checkpoint de probe ─────────────────────────
# lr=1e-4: mejor resultado en el experimento original IJEPA sin savgol (R²=0.358)

if [ ! -f "$BEST_CKPT" ]; then
    echo "WARN: No se encontró best_probe checkpoint en ${BEST_CKPT}. Abortando fine-tuning."
    exit 0
fi

SAVE="results/region_gadf2d_savgol/ft_gadf_jepa_savgol_${SLURM_JOB_ID}_ftlr1e-4/"

echo ""
echo "=========================================="
echo "Lanzando fine-tuning con: ${BEST_CKPT}"
echo "lr=1e-4  ->  ${SAVE}"
echo "=========================================="

python main_finetune_gadf2d.py \
    --data_path data/SoilDataset_NIR_gadf2d_savgol \
    --gadf_norm_stats data/gadf_norm_stats_v2_savgol.json \
    --region_tasks \
    --split test \
    --device cuda \
    --epoch 1000 \
    --batch_size 16 \
    --lr 1e-4 \
    --wd 1e-5 \
    --model_weight "${BEST_CKPT}" \
    --model_name vit_base \
    --patch_size 16 \
    --crop_size 224 \
    --patience 30 \
    --k_spt 25 \
    --k_qry 25 \
    --save_path "${SAVE}"

echo ""
echo "=========================================="
echo "Pipeline completo (pretrain + finetune lr=1e-4). Job $SLURM_JOB_ID done."
echo "=========================================="
