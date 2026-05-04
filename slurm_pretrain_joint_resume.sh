#!/bin/bash
#SBATCH --job-name=pretrain_joint_resume_ep600
#SBATCH --partition=dgx2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --output=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint/slurm_%j.out
#SBATCH --error=/mnt/homeGPU/igarzon/Meta-Learning/ijepa_spectra2d/logs/gadf_soil_nir_joint/slurm_%j.err
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
mkdir -p logs/gadf_soil_nir_joint

# -- Info del job
echo "=========================================="
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "Config:   configs/gadf_soil_nir_joint_resume.yaml"
echo "GPU:      cuda:0"
echo "Head:     IdentityNet (use_projection=false)"
echo "Proto:    smooth_l1  |  Lambda: 0.025"
echo "Resume:   ep400 -> ep600  (coseno reiniciado, Opcion A)"
echo "Checkpoint: logs/gadf_soil_nir_joint/gadf_joint_20260414_094939/gadf_joint-latest.pth.tar"
echo "=========================================="
echo ""
echo "NOTA: Al arrancar deben aparecer en el log:"
echo "  - 'loaded pretrained encoder from epoch 400' x3"
echo "  - Primera epoch impresa: 'Epoch 401 / 600' (o similar)"
echo "  Si aparece 'Epoch 1 / 600' => resume ha fallado, cancelar job."
echo "=========================================="

# -- Backup defensivo del params-joint.yaml original antes de que se sobrescriba
CKPT_DIR="logs/gadf_soil_nir_joint/gadf_joint_20260414_094939"
if [ -f "${CKPT_DIR}/params-joint.yaml" ] && [ ! -f "${CKPT_DIR}/params-joint.yaml.pre-resume" ]; then
    cp "${CKPT_DIR}/params-joint.yaml" "${CKPT_DIR}/params-joint.yaml.pre-resume"
    echo "Backup creado: ${CKPT_DIR}/params-joint.yaml.pre-resume"
fi

# -- Resume desde ep400, extension a 600 epocas con coseno reiniciado
# Todos los hiperparametros vienen del YAML; no se pasan CLI overrides para
# evitar desincronias entre CLI y lo que se loguea en params-joint.yaml.
python main_joint_pretrain.py \
    --fname configs/gadf_soil_nir_joint_resume.yaml \
    --devices cuda:0
