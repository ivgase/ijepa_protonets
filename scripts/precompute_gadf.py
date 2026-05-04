"""
Pre-computa imágenes GADF 224×224 para todos los espectros de X_supp.csv y X_query.csv
y las guarda en data/gadf_224.h5 con shape (N, 1, 224, 224), dtype float16.

Flags:
  --encode_diagonal   Codifica la magnitud espectral (normalizada globalmente)
                      en la diagonal de la GADF (por defecto desactivado).
  --stats_path        Ruta al JSON con min/max globales PAA (necesario si
                      --encode_diagonal está activo).
"""

import argparse
import os
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

import h5py
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField

# Añadir raíz del proyecto al path para importar src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = "/mnt/homeGPU/igarzon/Meta-Learning/SpectraI-JEPA/data/SoilDataset_NIR_agg"
CSV_FILES = ["X_supp.csv", "X_query.csv"]
_PROJECT_DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
IMAGE_SIZE = 224
BATCH_SIZE = 256


def main():
    parser = argparse.ArgumentParser(description="Pre-computa imágenes GADF")
    parser.add_argument("--encode_diagonal", action="store_true",
                        help="Codifica magnitud espectral en la diagonal")
    parser.add_argument("--stats_path",
                        default=os.path.join(_PROJECT_DATA, "gadf_paa_global_stats_v2.json"),
                        help="Ruta al JSON con min/max globales PAA")
    parser.add_argument("--output",
                        default=None,
                        help="Ruta de salida del HDF5 (por defecto: data/gadf_224_v2[_diagonal][_savgol].h5)")
    parser.add_argument("--savgol", action="store_true",
                        help="Aplica filtro Savitzky-Golay antes de la transformación GADF")
    parser.add_argument("--savgol_window", type=int, default=15,
                        help="Longitud de ventana SG (impar, default: 15)")
    parser.add_argument("--savgol_polyorder", type=int, default=2,
                        help="Orden del polinomio SG (default: 2)")
    parser.add_argument("--savgol_deriv", type=int, default=0,
                        help="Derivada SG: 0=suavizado, 1=primera derivada (default: 0)")
    args = parser.parse_args()

    suffix = ""
    if args.encode_diagonal:
        suffix += "_diagonal"
    if args.savgol:
        if args.savgol_deriv == 0:
            suffix += "_savgol"
        else:
            suffix += f"_savgol_d{args.savgol_deriv}"
    output_path = args.output or os.path.join(_PROJECT_DATA, f"gadf_224_v2{suffix}.h5")

    # Cargar stats globales si se usa diagonal
    global_min, global_max = None, None
    if args.encode_diagonal:
        from src.gadf_utils import encode_diagonal, load_global_stats
        global_min, global_max = load_global_stats(args.stats_path)
        print(f"Diagonal encoding ON (min={global_min:.4f}, max={global_max:.4f})")

    # Cargar todos los espectros
    dfs = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        print(f"Cargando {path} ...")
        df = pd.read_csv(path)
        print(f"  {df.shape[0]} muestras x {df.shape[1]} wavelengths")
        dfs.append(df)

    X = np.concatenate([df.values for df in dfs], axis=0).astype(np.float32)

    if args.savgol:
        from src.gadf_utils import apply_savitzky_golay
        X = apply_savitzky_golay(X, args.savgol_window, args.savgol_polyorder, args.savgol_deriv)
        print(f"Savitzky-Golay aplicado (window={args.savgol_window}, poly={args.savgol_polyorder}, deriv={args.savgol_deriv})")

    N = X.shape[0]
    print(f"\nTotal: {N} muestras x {X.shape[1]} wavelengths")

    # Crear directorio de salida
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    gaf = GramianAngularField(image_size=IMAGE_SIZE, method="difference")

    print(f"\nGenerando GADF {IMAGE_SIZE}x{IMAGE_SIZE} -> {output_path}")
    t0 = time.time()

    with h5py.File(output_path, "w") as f:
        dset = f.create_dataset(
            "images",
            shape=(N, 1, IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.float16,
            chunks=(64, 1, IMAGE_SIZE, IMAGE_SIZE),
        )

        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            batch = gaf.fit_transform(X[start:end])  # (B, 224, 224)
            if args.encode_diagonal:
                encode_diagonal(batch, X[start:end], global_min, global_max,
                                IMAGE_SIZE)
            dset[start:end, 0, :, :] = batch.astype(np.float16)

            if end % 1000 < BATCH_SIZE or end == N:
                elapsed = time.time() - t0
                print(f"  {end:>6d}/{N} muestras  ({elapsed:.1f}s)")

    total = time.time() - t0
    print(f"\nCompletado en {total:.1f}s")
    print(f"Archivo: {output_path}")
    print(f"Shape: ({N}, 1, {IMAGE_SIZE}, {IMAGE_SIZE}), dtype: float16")


if __name__ == "__main__":
    main()
