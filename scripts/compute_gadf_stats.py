"""
Calcula media y desviación estándar global de imágenes GADF 224×224
generadas a partir de las primeras 2000 muestras de X_supp.csv.
"""

import os

# Asegurar que se use la libstdc++ del conda env (tiene GLIBCXX_3.4.29)
_conda_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "metaenv_prueba", "lib")
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
from pyts.image import GramianAngularField

CSV_PATH = "/mnt/homeGPU/igarzon/Meta-Learning/SpectraMAENet/data/Soil_NIR_AGG/X_supp.csv"
N_SAMPLES = 10000
IMAGE_SIZE = 224

def main():
    print(f"Cargando primeras {N_SAMPLES} filas de {CSV_PATH} ...")
    X = pd.read_csv(CSV_PATH, index_col=0, nrows=N_SAMPLES).values
    print(f"  Shape: {X.shape}")

    gaf = GramianAngularField(image_size=IMAGE_SIZE, method="difference")

    print("Computando imágenes GADF ...")
    images = gaf.fit_transform(X)  # (N_SAMPLES, 224, 224)
    print(f"  Shape imágenes: {images.shape}")

    mean = np.mean(images).item()
    std = np.std(images).item()

    print()
    print(f"GADF_MEAN = ({mean:.4f},)")
    print(f"GADF_STD  = ({std:.4f},)")
    
    import matplotlib.pyplot as plt
    plt.imshow(images[0], cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Ejemplo de imagen GADF")
    # plt.show()
    plt.savefig("gadf_example.png")


if __name__ == "__main__":
    main()
