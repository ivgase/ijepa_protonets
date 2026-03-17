"""Shared GADF utilities: PAA resampling, diagonal encoding, global stats."""

import json
import numpy as np


def paa_resample(X, output_size=224):
    """Replicate pyts PAA (non-overlapping) resampling.

    Args:
        X: (N, L) array of raw spectra
        output_size: target length (default 224)
    Returns:
        X_paa: (N, output_size) array
    """
    n_samples, n_timestamps = X.shape
    bounds = np.linspace(0, n_timestamps, output_size + 1).astype(np.int64)
    start = bounds[:-1]
    end = bounds[1:]
    X_paa = np.empty((n_samples, output_size), dtype=X.dtype)
    for j in range(output_size):
        X_paa[:, j] = X[:, start[j]:end[j]].mean(axis=1)
    return X_paa


def encode_diagonal(gadf_images, raw_spectra, global_min, global_max,
                    image_size=224):
    """Overwrite GADF diagonal with globally-normalized PAA values.

    Args:
        gadf_images: (N, H, W) array -- modified in place
        raw_spectra: (N, L) raw spectral values (before PAA/scaling)
        global_min, global_max: dataset-wide PAA min/max (scalars)
        image_size: PAA output size (should match H, W)
    Returns:
        gadf_images with diagonal overwritten
    """
    X_paa = paa_resample(raw_spectra, output_size=image_size)
    X_norm = 2.0 * (X_paa - global_min) / (global_max - global_min) - 1.0
    X_norm = np.clip(X_norm, -1.0, 1.0)
    idx = np.arange(image_size)
    gadf_images[:, idx, idx] = X_norm.astype(gadf_images.dtype)
    return gadf_images


def load_global_stats(stats_path):
    """Load PAA global min/max from JSON.

    Returns:
        (global_min, global_max) tuple of floats
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    return stats['paa_global_min'], stats['paa_global_max']


def load_norm_stats(stats_path, use_diagonal=False):
    """Load GADF normalization mean/std from JSON.

    The JSON contains two variants: 'no_diagonal' and 'with_diagonal'.
    Selects automatically based on use_diagonal flag.

    Returns:
        ((mean,), (std,)) tuple suitable for transforms.Normalize
    """
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    key = 'with_diagonal' if use_diagonal else 'no_diagonal'
    entry = stats[key]
    if entry['gadf_mean'] is None or entry['gadf_std'] is None:
        raise ValueError(
            f"Norm stats for '{key}' not yet computed in {stats_path}. "
            f"Run: python scripts/compute_gadf_stats.py --from_h5"
        )
    return (entry['gadf_mean'],), (entry['gadf_std'],)
