"""
Pre-computa imágenes GADF 2D para las tareas downstream y las guarda en una
estructura espejo con tensores .pt en lugar de CSVs de espectros.

Estructura de salida:
    <dst>/
    ├── splits.csv                         (copia)
    ├── Soil_Austria-CaCO3/
    │   ├── X_supp.pt      (N, 1, 224, 224) float16
    │   ├── X_query.pt     (N, 1, 224, 224) float16
    │   ├── y_supp.csv     → symlink al original
    │   ├── y_query.csv    → symlink al original
    │   └── fixed_val_*    → symlinks al original
    └── ...

Cada X_supp.csv / X_query.csv se computa individualmente (sin deduplicación).

Flags:
    --encode_diagonal   Codifica magnitud espectral en la diagonal GADF
    --stats_path        Ruta al JSON con min/max globales PAA

Uso:
    python scripts/precompute_gadf_downstream.py \
        --src ../SpectraI-JEPA/data/Soil_NIR_AGG_mixed \
        --dst data/Soil_NIR_AGG_mixed_gadf2d \
        --image_size 224
"""

import argparse
import os
import shutil
import sys
import time

# Asegurar que se use la libstdc++ del conda env (tiene GLIBCXX_3.4.29)
_conda_lib = os.path.normpath(
    "/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib"
)
os.environ["LD_LIBRARY_PATH"] = (
    _conda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
)
import ctypes
ctypes.CDLL(os.path.join(_conda_lib, "libstdc++.so.6"))

import numpy as np
import pandas as pd
import torch
from pyts.image import GramianAngularField

# Añadir raíz del proyecto al path para importar src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Helpers
# ============================================================================

def compute_gadf(csv_path, image_size, batch_size=256,
                 global_min=None, global_max=None, savgol_params=None):
    """Lee un CSV de espectros y devuelve un tensor GADF (N, 1, H, W) float16.

    Si global_min/global_max se proporcionan, codifica la magnitud espectral
    normalizada globalmente en la diagonal de la GADF.
    Si savgol_params es un dict con window_length/polyorder/deriv, aplica
    filtro Savitzky-Golay antes de la transformación GADF.
    """
    df = pd.read_csv(csv_path, index_col=0)
    X = df.values.astype(np.float32)

    if savgol_params is not None:
        from src.gadf_utils import apply_savitzky_golay
        X = apply_savitzky_golay(X, **savgol_params)

    N = X.shape[0]

    gaf = GramianAngularField(image_size=image_size, method="difference")

    chunks = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = gaf.transform(X[start:end])  # (B, H, W)
        if global_min is not None and global_max is not None:
            from src.gadf_utils import encode_diagonal
            encode_diagonal(batch, X[start:end], global_min, global_max,
                            image_size)
        chunks.append(batch)

    images = np.concatenate(chunks, axis=0)  # (N, H, W)
    tensor = torch.from_numpy(images).unsqueeze(1).to(torch.float16)  # (N, 1, H, W)
    return tensor


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pre-computa GADF 2D para tareas downstream")
    parser.add_argument("--src", required=True,
                        help="Directorio fuente con subdirectorios de tareas")
    parser.add_argument("--dst", required=True,
                        help="Directorio de salida para la estructura GADF")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--encode_diagonal", action="store_true",
                        help="Codifica magnitud espectral en la diagonal")
    parser.add_argument("--stats_path",
                        default=os.path.join(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__))),
                            "data", "gadf_paa_global_stats.json"),
                        help="Ruta al JSON con min/max globales PAA")
    parser.add_argument("--savgol", action="store_true",
                        help="Aplica filtro Savitzky-Golay antes de la transformación GADF")
    parser.add_argument("--savgol_window", type=int, default=15,
                        help="Longitud de ventana SG (impar, default: 15)")
    parser.add_argument("--savgol_polyorder", type=int, default=2,
                        help="Orden del polinomio SG (default: 2)")
    parser.add_argument("--savgol_deriv", type=int, default=0,
                        help="Derivada SG: 0=suavizado, 1=primera derivada (default: 0)")
    args = parser.parse_args()

    # Cargar stats globales si se usa diagonal
    global_min, global_max = None, None
    if args.encode_diagonal:
        from src.gadf_utils import load_global_stats
        global_min, global_max = load_global_stats(args.stats_path)
        print(f"Diagonal encoding ON (min={global_min:.4f}, max={global_max:.4f})")

    savgol_params = None
    if args.savgol:
        savgol_params = dict(window_length=args.savgol_window,
                             polyorder=args.savgol_polyorder,
                             deriv=args.savgol_deriv)
        print(f"Savitzky-Golay ON (window={args.savgol_window}, poly={args.savgol_polyorder}, deriv={args.savgol_deriv})")

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    os.makedirs(dst, exist_ok=True)

    # -- Copiar splits.csv
    splits_src = os.path.join(src, "splits.csv")
    if os.path.exists(splits_src):
        shutil.copy2(splits_src, os.path.join(dst, "splits.csv"))
        print("Copied splits.csv")

    # -- Encontrar directorios de tareas
    task_names = sorted([
        d for d in os.listdir(src)
        if os.path.isdir(os.path.join(src, d))
    ])
    print(f"Found {len(task_names)} task directories\n")

    total_samples = 0
    total_files = 0
    t0 = time.time()

    for i, task_name in enumerate(task_names):
        task_src = os.path.join(src, task_name)
        task_dst = os.path.join(dst, task_name)
        os.makedirs(task_dst, exist_ok=True)

        print(f"[{i+1}/{len(task_names)}] {task_name}")

        # -- Computar GADF para X_supp y X_query
        for x_name in ["X_supp", "X_query"]:
            csv_path = os.path.join(task_src, f"{x_name}.csv")
            pt_path = os.path.join(task_dst, f"{x_name}.pt")

            if not os.path.exists(csv_path):
                print(f"  WARNING: {x_name}.csv not found, skipping")
                continue

            tensor = compute_gadf(csv_path, args.image_size, args.batch_size,
                                  global_min, global_max, savgol_params)
            torch.save(tensor, pt_path)
            total_samples += tensor.shape[0]
            total_files += 1
            print(f"  {x_name}: {tuple(tensor.shape)}")

        # -- Symlink y_supp.csv, y_query.csv
        for y_name in ["y_supp.csv", "y_query.csv"]:
            y_src = os.path.join(task_src, y_name)
            y_dst = os.path.join(task_dst, y_name)
            if os.path.exists(y_src):
                if os.path.exists(y_dst) or os.path.islink(y_dst):
                    os.remove(y_dst)
                os.symlink(y_src, y_dst)

        # -- Symlink fixed_val_* y fixed_split_*
        for fname in os.listdir(task_src):
            if fname.startswith("fixed_val_") or fname.startswith("fixed_split_"):
                f_src = os.path.join(task_src, fname)
                f_dst = os.path.join(task_dst, fname)
                if os.path.exists(f_dst) or os.path.islink(f_dst):
                    os.remove(f_dst)
                os.symlink(f_src, f_dst)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed:.1f}s")
    print(f"  Files computed: {total_files} ({total_samples} total samples)")
    print(f"  Task dirs:      {len(task_names)}")
    print(f"  Output:         {dst}")


if __name__ == "__main__":
    main()
