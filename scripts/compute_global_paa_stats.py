"""
Calcula las estadísticas globales (min/max) de los espectros resampleados con PAA
para la normalización de la diagonal GADF.

Genera data/gadf_paa_global_stats.json con:
  {"paa_global_min": float, "paa_global_max": float}

Uso:
  python scripts/compute_global_paa_stats.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

# Añadir raíz del proyecto al path para importar src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.gadf_utils import paa_resample

DATA_DIR = "/mnt/homeGPU/igarzon/Meta-Learning/SpectraI-JEPA/data/SoilDataset_NIR_agg"
CSV_FILES = ["X_supp.csv", "X_query.csv"]
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "data", "gadf_paa_global_stats_v2.json")
IMAGE_SIZE = 224


def main():
    dfs = []
    for fname in CSV_FILES:
        path = os.path.join(DATA_DIR, fname)
        print(f"Cargando {path} ...")
        df = pd.read_csv(path)
        print(f"  {df.shape[0]} muestras x {df.shape[1]} wavelengths")
        dfs.append(df)

    X = np.concatenate([df.values for df in dfs], axis=0).astype(np.float32)
    print(f"\nTotal: {X.shape[0]} muestras x {X.shape[1]} wavelengths")

    print(f"Aplicando PAA resampling a {IMAGE_SIZE} puntos ...")
    X_paa = paa_resample(X, output_size=IMAGE_SIZE)
    print(f"  X_paa shape: {X_paa.shape}")

    paa_min = float(X_paa.min())
    paa_max = float(X_paa.max())
    print(f"\n  paa_global_min: {paa_min:.6f}")
    print(f"  paa_global_max: {paa_max:.6f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    stats = {"paa_global_min": paa_min, "paa_global_max": paa_max}
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nGuardado en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
