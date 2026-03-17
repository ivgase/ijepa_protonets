"""
Calcula media y desviación estándar global de imágenes GADF 224×224
y guarda el resultado en data/gadf_norm_stats.json para que el resto
de scripts lo lean automáticamente.

Soporta dos modos:
  - Desde CSV (default): lee X_supp.csv y computa GADF on-the-fly
  - Desde HDF5 (--from_h5): lee directamente del HDF5 precomputado
"""

import argparse
import json
import os

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_CSV = "/mnt/homeGPU/igarzon/Meta-Learning/SpectraMAENet/data/Soil_NIR_AGG/X_supp.csv"
DEFAULT_H5 = os.path.join(PROJECT_DIR, "data", "gadf_224.h5")
DEFAULT_OUTPUT = os.path.join(PROJECT_DIR, "data", "gadf_norm_stats.json")
IMAGE_SIZE = 224


def compute_from_csv(csv_path, n_samples, image_size):
    """Computa stats leyendo espectros del CSV y generando GADF on-the-fly."""
    import pandas as pd
    from pyts.image import GramianAngularField

    print(f"Cargando primeras {n_samples} filas de {csv_path} ...")
    X = pd.read_csv(csv_path, index_col=0, nrows=n_samples).values
    print(f"  Shape: {X.shape}")

    gaf = GramianAngularField(image_size=image_size, method="difference")
    print("Computando imágenes GADF ...")
    images = gaf.fit_transform(X)  # (N, H, W)
    print(f"  Shape imágenes: {images.shape}")
    return images


def compute_from_h5(h5_path, n_samples):
    """Computa stats leyendo directamente del HDF5 precomputado."""
    import h5py

    print(f"Leyendo HDF5: {h5_path} ...")
    with h5py.File(h5_path, 'r') as f:
        ds = f['images']
        total = ds.shape[0]
        n = min(n_samples, total) if n_samples > 0 else total
        print(f"  Dataset shape: {ds.shape}, usando {n} muestras")
        images = ds[:n, 0, :, :]  # (N, H, W) — quitar canal
    return images.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Compute GADF normalization stats")
    parser.add_argument('--from_h5', action='store_true',
                        help='Compute from precomputed HDF5 instead of CSV')
    parser.add_argument('--h5_path', default=DEFAULT_H5,
                        help=f'HDF5 path (default: {DEFAULT_H5})')
    parser.add_argument('--csv_path', default=DEFAULT_CSV,
                        help=f'CSV path (default: {DEFAULT_CSV})')
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples (0=all for H5, default: 10000)')
    parser.add_argument('--output', default=DEFAULT_OUTPUT,
                        help=f'Output JSON path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--diagonal', action='store_true',
                        help='Save stats under "with_diagonal" key (else "no_diagonal")')
    args = parser.parse_args()

    if args.from_h5:
        images = compute_from_h5(args.h5_path, args.n_samples)
    else:
        images = compute_from_csv(args.csv_path, args.n_samples, IMAGE_SIZE)

    mean = np.mean(images).item()
    std = np.std(images).item()

    variant = 'with_diagonal' if args.diagonal else 'no_diagonal'
    print(f"\n[{variant}]")
    print(f"GADF_MEAN = {mean:.6f}")
    print(f"GADF_STD  = {std:.6f}")

    # Load existing JSON (or create skeleton) and update only our variant
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            all_stats = json.load(f)
    else:
        all_stats = {
            "no_diagonal": {"gadf_mean": None, "gadf_std": None},
            "with_diagonal": {"gadf_mean": None, "gadf_std": None},
        }

    all_stats[variant] = {"gadf_mean": round(mean, 6), "gadf_std": round(std, 6)}
    with open(args.output, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nGuardado en: {args.output} (variante '{variant}')")


if __name__ == "__main__":
    main()
