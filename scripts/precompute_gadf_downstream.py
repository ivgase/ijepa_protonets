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

Uso:
    python scripts/precompute_gadf_downstream.py \
        --src ../SpectraI-JEPA/data/Soil_NIR_AGG_mixed \
        --dst data/Soil_NIR_AGG_mixed_gadf2d \
        --image_size 224
"""

import argparse
import os
import shutil
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


# ============================================================================
# Helpers
# ============================================================================

def compute_gadf(csv_path, image_size, batch_size=256):
    """Lee un CSV de espectros y devuelve un tensor GADF (N, 1, H, W) float16."""
    df = pd.read_csv(csv_path, index_col=0)
    X = df.values.astype(np.float32)
    N = X.shape[0]

    gaf = GramianAngularField(image_size=image_size, method="difference")

    chunks = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = gaf.transform(X[start:end])  # (B, H, W)
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
    args = parser.parse_args()

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

            tensor = compute_gadf(csv_path, args.image_size, args.batch_size)
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
