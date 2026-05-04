import os
from logging import getLogger

import h5py
import numpy as np
import pandas as pd
import torch

logger = getLogger()


class GADFDataset(torch.utils.data.Dataset):
    """Dataset de imágenes GADF para pretraining I-JEPA.

    Soporta dos modos:
    - precomputed=True: lee desde HDF5 (shape (N, 1, 224, 224), dtype float16)
    - precomputed=False: computa GADF on-the-fly desde CSV de espectros
    """

    def __init__(
        self,
        h5_path=None,
        csv_path=None,
        image_size=224,
        precomputed=True,
        transform=None,
        global_stats=None,
        savgol_params=None
    ):
        self.precomputed = precomputed
        self.transform = transform
        self.image_size = image_size
        self.global_stats = global_stats  # (global_min, global_max) or None

        if precomputed:
            assert h5_path is not None, "h5_path requerido en modo precomputed"
            self.h5_path = h5_path
            self._h5 = None
            # Abrir temporalmente para leer longitud
            with h5py.File(h5_path, "r") as f:
                self._len = f["images"].shape[0]
            logger.info(f"GADFDataset (precomputed): {self._len} muestras desde {h5_path}")
        else:
            assert csv_path is not None, "csv_path requerido en modo on-the-fly"
            from pyts.image import GramianAngularField
            self.spectra = pd.read_csv(csv_path, index_col=0).values.astype(np.float32)
            if savgol_params is not None:
                from src.gadf_utils import apply_savitzky_golay
                self.spectra = apply_savitzky_golay(self.spectra, **savgol_params)
            self._len = len(self.spectra)
            self.gaf = GramianAngularField(image_size=image_size, method="difference")
            logger.info(f"GADFDataset (on-the-fly): {self._len} muestras desde {csv_path}")

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.precomputed:
            # Apertura diferida: compatible con DataLoader multiprocess
            if self._h5 is None:
                self._h5 = h5py.File(self.h5_path, "r")
            img = self._h5["images"][idx].astype(np.float32)  # (1, 224, 224)
            img = torch.from_numpy(img)
        else:
            spectrum = self.spectra[idx : idx + 1]  # (1, n_wavelengths)
            gadf_img = self.gaf.transform(spectrum)  # (1, image_size, image_size)
            if self.global_stats is not None:
                from src.gadf_utils import encode_diagonal
                encode_diagonal(gadf_img, spectrum,
                                *self.global_stats, self.image_size)
            img = torch.from_numpy(gadf_img[0].astype(np.float32)).unsqueeze(0)  # (1, H, W)

        if self.transform is not None:
            img = self.transform(img)

        return img, 0


def make_gadf(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    h5_path=None,
    drop_last=True,
    **kwargs
):
    dataset = GADFDataset(
        h5_path=h5_path,
        precomputed=True,
        transform=transform)

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)

    logger.info("GADF unsupervised data loader created")

    return dataset, data_loader, dist_sampler


if __name__ == "__main__":
    import os
    import sys

    # Asegurar libstdc++ del conda env
    _conda_lib = "/mnt/homeGPU/igarzon/Meta-Learning/metaenv_prueba/lib"
    os.environ["LD_LIBRARY_PATH"] = _conda_lib + ":" + os.environ.get("LD_LIBRARY_PATH", "")
    import ctypes
    ctypes.CDLL(os.path.join(_conda_lib, "libstdc++.so.6"))

    from torchvision.transforms import Normalize

    H5_PATH = "data/gadf_224.h5"
    CSV_PATH = "/mnt/homeGPU/igarzon/Meta-Learning/SpectraMAENet/data/Soil_NIR_AGG/X_supp.csv"

    # --- Test precomputed mode ---
    print("=== Modo precomputed (HDF5) ===")
    ds_pre = GADFDataset(h5_path=H5_PATH, precomputed=True)
    img, label = ds_pre[0]
    print(f"  len(dataset): {len(ds_pre)}")
    print(f"  img shape: {img.shape}, dtype: {img.dtype}")
    print(f"  min: {img.min():.4f}, max: {img.max():.4f}, label: {label}")

    # Test DataLoader con num_workers=2
    print("\n  DataLoader (batch=4, workers=2):")
    dl = torch.utils.data.DataLoader(ds_pre, batch_size=4, num_workers=2)
    batch_imgs, batch_labels = next(iter(dl))
    print(f"    batch shape: {batch_imgs.shape}, labels: {batch_labels.tolist()}")

    # Test con transform
    print("\n  Con Normalize((0.0,), (0.5,)):")
    ds_norm = GADFDataset(h5_path=H5_PATH, precomputed=True,
                          transform=Normalize((0.0,), (0.5,)))
    img_norm, _ = ds_norm[0]
    print(f"    min: {img_norm.min():.4f}, max: {img_norm.max():.4f}")

    # --- Test on-the-fly mode ---
    print("\n=== Modo on-the-fly (CSV) ===")
    ds_fly = GADFDataset(csv_path=CSV_PATH, precomputed=False, image_size=224)
    img_fly, label_fly = ds_fly[0]
    print(f"  len(dataset): {len(ds_fly)}")
    print(f"  img shape: {img_fly.shape}, dtype: {img_fly.dtype}")
    print(f"  min: {img_fly.min():.4f}, max: {img_fly.max():.4f}, label: {label_fly}")

    # --- Comparar shapes ---
    print(f"\n=== Comparación ===")
    print(f"  precomputed shape: {img.shape}")
    print(f"  on-the-fly  shape: {img_fly.shape}")
    assert img.shape == img_fly.shape, "SHAPES NO COINCIDEN!"
    print("  OK: shapes coinciden")
