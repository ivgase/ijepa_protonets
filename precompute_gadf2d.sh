#!/bin/bash
# Precomputa imágenes GADF 2D para pre-training y tareas downstream.
#
# Uso:
#   bash precompute_gadf2d.sh                              # sin diagonal, sin SG
#   bash precompute_gadf2d.sh --diagonal                   # con diagonal
#   bash precompute_gadf2d.sh --savgol                     # SG suavizado (deriv=0) → _savgol
#   bash precompute_gadf2d.sh --savgol --savgol_deriv 1    # SG primera derivada   → _savgol_d1
#   bash precompute_gadf2d.sh --diagonal --savgol          # diagonal + SG
#   bash precompute_gadf2d.sh --diagonal --savgol --savgol_deriv 1

set -euo pipefail

# ── Configuración ─────────────────────────────────────────────────────────────
PYTHON=/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/bin/python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
SRC_TASKS="/mnt/homeGPU/igarzon/Meta-Learning/SpectraI-JEPA/data/SoilDataset_NIR"

# Parámetros Savitzky-Golay (defaults)
SAVGOL_WINDOW=15
SAVGOL_POLYORDER=2
SAVGOL_DERIV=0

# Parsear flags (incluyendo --savgol_deriv N)
DIAGONAL=false
SAVGOL=false
_next_is_deriv=false
for arg in "$@"; do
    if $_next_is_deriv; then
        SAVGOL_DERIV="$arg"
        _next_is_deriv=false
    elif [[ "$arg" == "--diagonal" ]]; then
        DIAGONAL=true
    elif [[ "$arg" == "--savgol" ]]; then
        SAVGOL=true
    elif [[ "$arg" == "--savgol_deriv" ]]; then
        _next_is_deriv=true
    fi
done

# Construir sufijos y flags extra para los scripts Python
DIAG_SUFFIX=""
SAVGOL_SUFFIX=""
DIAG_FLAGS=""
SAVGOL_FLAGS=""
$DIAGONAL && DIAG_SUFFIX="_diagonal" && DIAG_FLAGS="--encode_diagonal"
if $SAVGOL; then
    # Sufijo: _savgol para deriv=0, _savgol_d<N> para deriv>0
    if [[ "$SAVGOL_DERIV" == "0" ]]; then
        SAVGOL_SUFFIX="_savgol"
    else
        SAVGOL_SUFFIX="_savgol_d${SAVGOL_DERIV}"
    fi
    SAVGOL_FLAGS="--savgol --savgol_window $SAVGOL_WINDOW --savgol_polyorder $SAVGOL_POLYORDER --savgol_deriv $SAVGOL_DERIV"
fi

H5_OUT="$DATA_DIR/gadf_224_v2${DIAG_SUFFIX}${SAVGOL_SUFFIX}.h5"
DST_TASKS="$DATA_DIR/SoilDataset_NIR_gadf2d${DIAG_SUFFIX}${SAVGOL_SUFFIX}"
NORM_STATS="$DATA_DIR/gadf_norm_stats_v2.json"
PAA_STATS="$DATA_DIR/gadf_paa_global_stats_v2.json"

if $DIAGONAL; then
    NORM_KEY="with_diagonal"
    echo "=== Modo: CON diagonal ==="
else
    NORM_KEY="no_diagonal"
    echo "=== Modo: SIN diagonal ==="
fi
$SAVGOL && echo "=== Savitzky-Golay: ON (window=$SAVGOL_WINDOW, poly=$SAVGOL_POLYORDER, deriv=$SAVGOL_DERIV) ===" || true
echo ""

# ── Helpers ───────────────────────────────────────────────────────────────────
STEP=0
log_step() {
    STEP=$((STEP + 1))
    echo "──────────────────────────────────────────────────────────────"
    echo "  Paso $STEP: $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "──────────────────────────────────────────────────────────────"
}

check_file() {
    local path="$1"
    local desc="$2"
    if [[ ! -f "$path" ]]; then
        echo "  ERROR: no se generó el archivo esperado: $path" >&2
        exit 1
    fi
    local size
    size=$(du -sh "$path" | cut -f1)
    echo "  OK  $desc → $path  ($size)"
}

check_dir() {
    local path="$1"
    local desc="$2"
    if [[ ! -d "$path" ]]; then
        echo "  ERROR: no se creó el directorio esperado: $path" >&2
        exit 1
    fi
    local n
    n=$(find "$path" -name "X_supp.pt" | wc -l)
    echo "  OK  $desc → $path  ($n tareas con X_supp.pt)"
    if [[ "$n" -eq 0 ]]; then
        echo "  WARN: no se encontraron ficheros X_supp.pt en $path" >&2
        exit 1
    fi
}

check_json_key() {
    local path="$1"
    local key="$2"
    if ! python3 -c "
import json, sys
d = json.load(open('$path'))
assert '$key' in d, 'clave ausente'
v = d['$key']
assert isinstance(v, dict), 'valor no es dict'
assert v.get('gadf_mean') is not None and v.get('gadf_std') is not None, 'gadf_mean/gadf_std son null'
" 2>/dev/null; then
        echo "  ERROR: $path — la clave '$key' falta o tiene gadf_mean/gadf_std null" >&2
        exit 1
    fi
    local mean std
    mean=$(python3 -c "import json; print(json.load(open('$path'))['$key']['gadf_mean'])")
    std=$(python3  -c "import json; print(json.load(open('$path'))['$key']['gadf_std'])")
    echo "  OK  $path  (clave '$key': mean=$mean, std=$std)"
}

cd "$SCRIPT_DIR"

# ── Paso 1 (solo diagonal): calcular PAA min/max globales ─────────────────────
if $DIAGONAL; then
    log_step "Calcular estadísticas PAA globales → $PAA_STATS"
    $PYTHON scripts/compute_global_paa_stats.py
    check_file "$PAA_STATS" "gadf_paa_global_stats_v2.json"
    echo ""
fi

# ── Paso 2: generar HDF5 de pre-training ──────────────────────────────────────
log_step "Generar HDF5 de pre-training → $H5_OUT"
$PYTHON scripts/precompute_gadf.py \
    ${DIAG_FLAGS:+$DIAG_FLAGS} \
    ${DIAG_FLAGS:+--stats_path "$PAA_STATS"} \
    ${SAVGOL_FLAGS:+$SAVGOL_FLAGS}
check_file "$H5_OUT" "HDF5 pre-training"
echo ""

# ── Paso 3: calcular media/std para normalización ─────────────────────────────
log_step "Calcular estadísticas de normalización → $NORM_STATS  (clave '$NORM_KEY')"
NORM_FLAG=""
$DIAGONAL && NORM_FLAG="--diagonal"
$PYTHON scripts/compute_gadf_stats.py \
    --from_h5 --h5_path "$H5_OUT" \
    --output "$NORM_STATS" \
    $NORM_FLAG
check_file "$NORM_STATS" "gadf_norm_stats_v2.json"
check_json_key "$NORM_STATS" "$NORM_KEY"
echo ""

# ── Paso 4: generar GADF 2D para tareas downstream ────────────────────────────
log_step "Generar GADF 2D para tareas downstream → $DST_TASKS"
$PYTHON scripts/precompute_gadf_downstream.py \
    --src "$SRC_TASKS" \
    --dst "$DST_TASKS" \
    ${DIAG_FLAGS:+$DIAG_FLAGS} \
    ${DIAG_FLAGS:+--stats_path "$PAA_STATS"} \
    ${SAVGOL_FLAGS:+$SAVGOL_FLAGS}
check_dir "$DST_TASKS" "directorio de tareas downstream"
echo ""

# ── Resumen ───────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════════════════════"
echo "  COMPLETADO  $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "  HDF5 pre-training : $H5_OUT"
echo "  Norm stats        : $NORM_STATS"
echo "  Tareas downstream : $DST_TASKS"
$DIAGONAL && echo "  PAA stats         : $PAA_STATS"
$SAVGOL   && echo "  Savitzky-Golay    : window=$SAVGOL_WINDOW, poly=$SAVGOL_POLYORDER, deriv=$SAVGOL_DERIV"
echo "══════════════════════════════════════════════════════════════"
