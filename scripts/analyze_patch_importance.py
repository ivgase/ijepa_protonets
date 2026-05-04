"""Analyze per-patch importance maps grouped by soil property.

Reads the output of per_patch_probe.py (run on the --split train partition)
and computes:
  1. Property-level importance maps  — mean per-patch R² across all train
     countries for each soil property.
  2. Cross-country consistency       — Spearman ρ between countries within
     the same property (measures whether importance is truly property-specific).
  3. Cross-property distinctiveness  — Spearman ρ between property maps
     (measures whether different properties activate different spectral regions).

GADF structure context
----------------------
GADF(i,j) = cos(φ_i + φ_j).  The 14×14 patch grid (patch_size=16) maps to:
  • Diagonal patches (row==col) → individual wavelength bands.
  • Off-diagonal patches        → pairwise wavelength interactions.
Wavelength labels on the output figures use the formula:
  wl_center(p) = WL_START + (p * PATCH_SIZE + PATCH_SIZE//2) * (WL_END - WL_START)
                 / (GRID_SIZE * PATCH_SIZE - 1)

Outputs (all in --save_dir)
---------------------------
  importance_maps.npz            keys=property names, each ndarray [196]
  consistency_results.json
  property_distinctiveness.json
  heatmap_{prop}.png  (one per property)
  consistency_{prop}.png  (one per property — country K×K Spearman matrix)
  cross_property_correlation.png

Usage
-----
  python scripts/analyze_patch_importance.py \\
      --results_csv results/per_patch_probe_train/patch_probe_per_task.csv \\
      --save_dir    results/importance_analysis/
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE   = 14
PATCH_SIZE  = 16
N_PATCHES   = GRID_SIZE * GRID_SIZE  # 196
WL_START    = 401    # nm
WL_END      = 2499   # nm

PROPERTY_ORDER = ['CaCO3', 'Clay', 'OC', 'N', 'CEC', 'pH_h2o', 'ph_CaCl2']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_task(task_name: str):
    """Split 'Soil_{Country}-{Property}' → (country, property).

    Uses maxsplit=1 so that property names with hyphens (ph_CaCl2) and
    country names with spaces (United Kingdom) are handled correctly.
    """
    rest = task_name.replace('Soil_', '', 1)
    country, prop = rest.split('-', 1)
    return country, prop


def patch_wl_center(idx: int) -> int:
    """Return the centre wavelength (nm) for patch row/col index *idx*."""
    pixel_center = idx * PATCH_SIZE + PATCH_SIZE // 2
    wl = WL_START + pixel_center * (WL_END - WL_START) / (GRID_SIZE * PATCH_SIZE - 1)
    return int(round(wl))


def wl_tick_labels():
    """14 wavelength labels (nm) for the 14 patch positions along one axis."""
    return [str(patch_wl_center(i)) for i in range(GRID_SIZE)]


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def load_per_task_r2(results_csv: str) -> dict:
    """Load patch_probe_per_task.csv and return {task_name: ndarray[196]}
    where values are the mean R² across repeats.
    """
    df = pd.read_csv(results_csv)
    # mean over repeats per (task, patch_idx)
    agg = df.groupby(['task', 'patch_idx'])['r2'].mean().reset_index()
    task_r2 = {}
    for task_name, grp in agg.groupby('task'):
        grp_sorted = grp.sort_values('patch_idx')
        if len(grp_sorted) != N_PATCHES:
            print(f'  WARNING: {task_name} has {len(grp_sorted)} patches '
                  f'(expected {N_PATCHES}), skipping')
            continue
        task_r2[task_name] = grp_sorted['r2'].values.astype(np.float32)
    return task_r2


def compute_property_maps(task_r2: dict, min_countries: int = 1):
    """Group per-task R² maps by property and compute per-property mean.

    Returns
    -------
    prop_r2   : {prop: ndarray[196]}  mean R² across countries
    prop_tasks: {prop: list[task_name]}
    """
    # group tasks by property
    by_prop: dict[str, list] = {}
    for tname, r2 in task_r2.items():
        _, prop = parse_task(tname)
        by_prop.setdefault(prop, []).append((tname, r2))

    prop_r2    = {}
    prop_tasks = {}
    for prop, items in by_prop.items():
        if len(items) < min_countries:
            print(f'  Skipping {prop}: only {len(items)} countries '
                  f'(min_countries={min_countries})')
            continue
        stacked = np.stack([r2 for _, r2 in items], axis=0)  # [K, 196]
        prop_r2[prop]    = np.nanmean(stacked, axis=0)
        prop_tasks[prop] = [t for t, _ in items]
    return prop_r2, prop_tasks


def compute_cross_country_consistency(task_r2: dict):
    """For each property, compute Spearman ρ between all pairs of countries.

    Returns
    -------
    consistency: {prop: {'countries': list, 'rho_matrix': ndarray[K,K],
                          'mean_rho': float, 'std_rho': float}}
    """
    # group by property
    by_prop: dict[str, dict] = {}
    for tname, r2 in task_r2.items():
        country, prop = parse_task(tname)
        by_prop.setdefault(prop, {})[country] = r2

    consistency = {}
    for prop, country_map in by_prop.items():
        countries = sorted(country_map.keys())
        K = len(countries)
        if K < 2:
            continue
        mat = np.stack([country_map[c] for c in countries], axis=0)  # [K, 196]

        rho_mat = np.full((K, K), np.nan)
        for i in range(K):
            for j in range(K):
                if i == j:
                    rho_mat[i, j] = 1.0
                else:
                    # Only use finite values
                    mask = np.isfinite(mat[i]) & np.isfinite(mat[j])
                    if mask.sum() >= 5:
                        rho, _ = spearmanr(mat[i][mask], mat[j][mask])
                        rho_mat[i, j] = rho

        upper = rho_mat[np.triu_indices(K, k=1)]
        valid = upper[np.isfinite(upper)]
        consistency[prop] = {
            'countries':  countries,
            'rho_matrix': rho_mat,
            'mean_rho':   float(np.nanmean(valid)) if len(valid) else float('nan'),
            'std_rho':    float(np.nanstd(valid))  if len(valid) else float('nan'),
            'n_countries': K,
        }
    return consistency


def compute_cross_property_distinctiveness(prop_r2: dict):
    """Spearman ρ between all pairs of property importance maps.

    Returns
    -------
    props  : list[str]
    rho_mat: ndarray[P, P]
    """
    props = [p for p in PROPERTY_ORDER if p in prop_r2]
    # add any props not in PROPERTY_ORDER (shouldn't happen but be safe)
    for p in sorted(prop_r2.keys()):
        if p not in props:
            props.append(p)

    P = len(props)
    rho_mat = np.full((P, P), np.nan)
    for i in range(P):
        for j in range(P):
            if i == j:
                rho_mat[i, j] = 1.0
            else:
                vi = prop_r2[props[i]]
                vj = prop_r2[props[j]]
                mask = np.isfinite(vi) & np.isfinite(vj)
                if mask.sum() >= 5:
                    rho, _ = spearmanr(vi[mask], vj[mask])
                    rho_mat[i, j] = rho
    return props, rho_mat


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def _heatmap_colormap_and_vrange(data: np.ndarray):
    """Choose colormap and vmin/vmax for a patch importance heatmap."""
    finite = data[np.isfinite(data)]
    if len(finite) == 0:
        return 'RdYlGn', 0.0, 1.0
    v2  = np.percentile(finite, 2)
    v98 = np.percentile(finite, 98)
    if v2 < 0:
        # symmetric around 0
        vabs = max(abs(v2), abs(v98))
        return 'RdYlGn', -vabs, vabs
    return 'YlGn', v2, v98


def plot_property_heatmap(prop: str, r2_map: np.ndarray, n_countries: int,
                          save_path: str):
    """14×14 heatmap of mean R² for one property, axes labelled in nm."""
    grid = r2_map.reshape(GRID_SIZE, GRID_SIZE)
    cmap, vmin, vmax = _heatmap_colormap_and_vrange(r2_map)

    wl_labels = wl_tick_labels()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='equal')
    plt.colorbar(im, ax=ax, label='Mean R²')

    ax.set_xticks(range(GRID_SIZE))
    ax.set_xticklabels(wl_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(GRID_SIZE))
    ax.set_yticklabels(wl_labels, fontsize=6)
    ax.set_xlabel('Column wavelength (nm)')
    ax.set_ylabel('Row wavelength (nm)')
    ax.set_title(f'{prop}  —  mean per-patch R²  ({n_countries} train countries)')

    # Mark diagonal patches (individual wavelength bands)
    for d in range(GRID_SIZE):
        rect = plt.Rectangle((d - 0.5, d - 0.5), 1, 1,
                              linewidth=1.5, edgecolor='blue',
                              facecolor='none')
        ax.add_patch(rect)

    # Annotate best patch
    best_idx = int(np.nanargmax(r2_map))
    br, bc = best_idx // GRID_SIZE, best_idx % GRID_SIZE
    ax.add_patch(plt.Rectangle((bc - 0.5, br - 0.5), 1, 1,
                                linewidth=2, edgecolor='gold',
                                facecolor='none'))
    ax.text(bc, br, f'{r2_map[best_idx]:.2f}', ha='center', va='center',
            fontsize=5, fontweight='bold', color='gold')

    fig.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_consistency_matrix(prop: str, countries: list,
                            rho_mat: np.ndarray, mean_rho: float,
                            save_path: str):
    """K×K Spearman ρ matrix between countries for a single property."""
    K = len(countries)
    fig, ax = plt.subplots(figsize=(max(4, K * 0.6 + 1),
                                    max(4, K * 0.6 + 1)))
    finite = rho_mat[np.isfinite(rho_mat)]
    vmin = np.nanmin(finite) if len(finite) else -1
    vmax = 1.0
    im = ax.imshow(rho_mat, cmap='RdYlGn', vmin=vmin, vmax=vmax,
                   interpolation='nearest', aspect='equal')
    plt.colorbar(im, ax=ax, label='Spearman ρ')

    ax.set_xticks(range(K))
    ax.set_xticklabels(countries, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(K))
    ax.set_yticklabels(countries, fontsize=7)
    ax.set_title(f'{prop}  —  cross-country Spearman ρ\n'
                 f'mean (upper triangle) = {mean_rho:.3f}')

    for i in range(K):
        for j in range(K):
            v = rho_mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=6, color='black')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_cross_property_correlation(props: list, rho_mat: np.ndarray,
                                    save_path: str):
    """7×7 (or P×P) Spearman ρ between property importance maps."""
    P = len(props)
    fig, ax = plt.subplots(figsize=(max(5, P * 0.8 + 1),
                                    max(5, P * 0.8 + 1)))
    im = ax.imshow(rho_mat, cmap='RdYlGn', vmin=-1, vmax=1,
                   interpolation='nearest', aspect='equal')
    plt.colorbar(im, ax=ax, label='Spearman ρ')

    ax.set_xticks(range(P))
    ax.set_xticklabels(props, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(P))
    ax.set_yticklabels(props, fontsize=8)
    ax.set_title('Cross-property distinctiveness\n'
                 '(Spearman ρ between property importance maps)')

    for i in range(P):
        for j in range(P):
            v = rho_mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color='black')

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    p = argparse.ArgumentParser('Patch importance analysis', add_help=False)
    p.add_argument('--results_csv',
                   default='results/per_patch_probe_joint_ep700_train/patch_probe_per_task.csv',
                   type=str,
                   help='patch_probe_per_task.csv (train or test split)')
    p.add_argument('--save_dir',
                   default='results/importance_analysis_joint_ep700/',
                   type=str)
    p.add_argument('--min_countries', type=int, default=1,
                   help='Minimum countries per property to include in analysis')
    p.add_argument('--grid_size', type=int, default=GRID_SIZE)
    p.add_argument('--wl_start', type=int, default=WL_START)
    p.add_argument('--wl_end', type=int, default=WL_END)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # Override module-level constants from CLI (for non-standard configs)
    global GRID_SIZE, N_PATCHES, WL_START, WL_END
    GRID_SIZE = args.grid_size
    N_PATCHES = GRID_SIZE * GRID_SIZE
    WL_START  = args.wl_start
    WL_END    = args.wl_end

    os.makedirs(args.save_dir, exist_ok=True)
    print(f'Loading {args.results_csv} ...')

    task_r2 = load_per_task_r2(args.results_csv)
    print(f'  Loaded {len(task_r2)} tasks')

    # -----------------------------------------------------------------------
    # 1. Property importance maps
    # -----------------------------------------------------------------------
    print('\n--- Property importance maps ---')
    prop_r2, prop_tasks = compute_property_maps(
        task_r2, min_countries=args.min_countries)

    for prop, r2 in prop_r2.items():
        n_c = len(prop_tasks[prop])
        best = int(np.nanargmax(r2))
        br, bc = best // GRID_SIZE, best % GRID_SIZE
        print(f'  {prop:12s}: {n_c} countries, '
              f'mean R²={np.nanmean(r2):.4f}, '
              f'best patch ({br},{bc}) @ {patch_wl_center(br)}×'
              f'{patch_wl_center(bc)} nm = {r2[best]:.4f}')

    # Save importance maps
    np.savez(os.path.join(args.save_dir, 'importance_maps.npz'), **prop_r2)
    print(f'\nSaved importance_maps.npz  ({list(prop_r2.keys())})')

    # -----------------------------------------------------------------------
    # 2. Cross-country consistency
    # -----------------------------------------------------------------------
    print('\n--- Cross-country consistency (Spearman ρ) ---')
    consistency = compute_cross_country_consistency(task_r2)

    consistency_out = {}
    for prop, res in consistency.items():
        print(f'  {prop:12s}: {res["n_countries"]} countries, '
              f'mean ρ = {res["mean_rho"]:.3f} ± {res["std_rho"]:.3f}')
        consistency_out[prop] = {
            'countries':   res['countries'],
            'mean_rho':    res['mean_rho'],
            'std_rho':     res['std_rho'],
            'n_countries': res['n_countries'],
            'pairwise_rho': {
                f'{res["countries"][i]}_vs_{res["countries"][j]}':
                    float(res['rho_matrix'][i, j])
                for i in range(res['n_countries'])
                for j in range(i + 1, res['n_countries'])
                if np.isfinite(res['rho_matrix'][i, j])
            }
        }

    with open(os.path.join(args.save_dir, 'consistency_results.json'), 'w') as f:
        json.dump(consistency_out, f, indent=2)

    # -----------------------------------------------------------------------
    # 3. Cross-property distinctiveness
    # -----------------------------------------------------------------------
    print('\n--- Cross-property distinctiveness ---')
    props, cross_rho = compute_cross_property_distinctiveness(prop_r2)

    cross_out = {
        'properties': props,
        'rho_matrix': cross_rho.tolist(),
    }
    with open(os.path.join(args.save_dir,
                           'property_distinctiveness.json'), 'w') as f:
        json.dump(cross_out, f, indent=2)

    for i, pi in enumerate(props):
        for j, pj in enumerate(props):
            if j > i and np.isfinite(cross_rho[i, j]):
                print(f'  {pi:12s} vs {pj:12s}: ρ = {cross_rho[i,j]:.3f}')

    # -----------------------------------------------------------------------
    # 4. Visualizations
    # -----------------------------------------------------------------------
    print('\n--- Generating figures ---')

    # Property heatmaps
    for prop, r2 in prop_r2.items():
        n_c = len(prop_tasks[prop])
        out = os.path.join(args.save_dir, f'heatmap_{prop}.png')
        plot_property_heatmap(prop, r2, n_c, out)
        print(f'  heatmap_{prop}.png')

    # Cross-country consistency matrices
    for prop, res in consistency.items():
        out = os.path.join(args.save_dir, f'consistency_{prop}.png')
        plot_consistency_matrix(
            prop, res['countries'], res['rho_matrix'],
            res['mean_rho'], out)
        print(f'  consistency_{prop}.png')

    # Cross-property distinctiveness
    out = os.path.join(args.save_dir, 'cross_property_correlation.png')
    plot_cross_property_correlation(props, cross_rho, out)
    print('  cross_property_correlation.png')

    print(f'\nAll results saved to {args.save_dir}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
