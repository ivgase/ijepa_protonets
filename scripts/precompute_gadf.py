"""
Pre-computa imágenes GADF 224×224 para todos los espectros de X_supp.csv y X_query.csv
y las guarda en data/gadf_224.h5 con shape (N, 1, 224, 224), dtype float16.
"""

import os
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

DATA_DIR = "/mnt/homeGPU/igarzon/Meta-Learning/SpectraMAENet/data/Soil_NIR_AGG"
CSV_FILES = ["X_supp.csv", "X_query.csv"]
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "gadf_224.h5")
IMAGE_SIZE = 224
BATCH_SIZE = 256


def main():
    # Cargar todos los espectros
    dfs = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        print(f"Cargando {path} ...")
        df = pd.read_csv(path, index_col=0)
        print(f"  {df.shape[0]} muestras x {df.shape[1]} wavelengths")
        dfs.append(df)

    X = np.concatenate([df.values for df in dfs], axis=0).astype(np.float32)
    N = X.shape[0]
    print(f"\nTotal: {N} muestras x {X.shape[1]} wavelengths")

    # Crear directorio de salida
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    gaf = GramianAngularField(image_size=IMAGE_SIZE, method="difference")

    print(f"\nGenerando GADF {IMAGE_SIZE}x{IMAGE_SIZE} -> {OUTPUT_PATH}")
    t0 = time.time()

    with h5py.File(OUTPUT_PATH, "w") as f:
        dset = f.create_dataset(
            "images",
            shape=(N, 1, IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.float16,
            chunks=(64, 1, IMAGE_SIZE, IMAGE_SIZE),
        )

        for start in range(0, N, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N)
            batch = gaf.fit_transform(X[start:end])  # (B, 224, 224)
            dset[start:end, 0, :, :] = batch.astype(np.float16)

            if end % 1000 < BATCH_SIZE or end == N:
                elapsed = time.time() - t0
                print(f"  {end:>6d}/{N} muestras  ({elapsed:.1f}s)")

    total = time.time() - t0
    print(f"\nCompletado en {total:.1f}s")
    print(f"Archivo: {OUTPUT_PATH}")
    print(f"Shape: ({N}, 1, {IMAGE_SIZE}, {IMAGE_SIZE}), dtype: float16")


if __name__ == "__main__":
    main()
