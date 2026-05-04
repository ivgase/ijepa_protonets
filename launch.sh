#!/bin/bash
# Wrapper para sbatch que registra el objetivo del experimento en experiments.log
#
# Uso: ./launch.sh <slurm_script.sh> [args adicionales para sbatch]

set -e

# Detectar --after <JOBID> para dependencia SLURM
DEPENDENCY=""
if [ "$1" = "--after" ]; then
    if [ -z "$2" ]; then
        echo "Error: --after requiere un job ID."
        exit 1
    fi
    DEPENDENCY="--dependency=afterok:$2"
    shift 2
fi

SCRIPT="$1"
if [ -z "$SCRIPT" ]; then
    echo "Uso: $0 [--after <JOBID>] <slurm_script.sh> [args...]"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "Error: no existe '$SCRIPT'"
    exit 1
fi

shift  # el resto: args para sbatch antes de '--', args para el script después de '--'

SBATCH_ARGS=()
SCRIPT_ARGS=()
FOUND_SEP=false
for arg in "$@"; do
    if [ "$arg" = "--" ]; then
        FOUND_SEP=true
    elif $FOUND_SEP; then
        SCRIPT_ARGS+=("$arg")
    else
        SBATCH_ARGS+=("$arg")
    fi
done

# Pedir objetivo
echo ""
echo "Objetivo de este experimento (una línea):"
read -r OBJETIVO
if [ -z "$OBJETIVO" ]; then
    echo "Error: el objetivo no puede estar vacío."
    exit 1
fi

# Lanzar
OUTPUT=$(sbatch $DEPENDENCY "${SBATCH_ARGS[@]}" "$SCRIPT" "${SCRIPT_ARGS[@]}")
JOB_ID=$(echo "$OUTPUT" | awk '{print $NF}')

# Registrar
LOG="experiments.log"
TIMESTAMP=$(date "+%Y-%m-%d %H:%M")
echo "${TIMESTAMP} | job=${JOB_ID} | script=$(basename $SCRIPT) | ${OBJETIVO}" >> "$LOG"

echo "$OUTPUT"
echo "Registrado en $LOG"
