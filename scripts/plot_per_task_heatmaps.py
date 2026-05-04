"""Plot per-task 14×14 R² heatmaps from patch_probe_per_task.csv.

Reads existing results (no re-running needed) and generates:
  - One figure per soil property, with regions as columns.
  - A summary figure comparing the mean R² profile per property.

Usage:
  python scripts/plot_per_task_heatmaps.py \
      --results_dir results/per_patch_probe/ \
      --save_dir results/per_patch_probe/per_task_heatmaps/
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GRID_SIZE = 14
N_PATCHES = GRID_SIZE * GRID_SIZE

PROPERTY_ORDER = ['CaCO3', 'Clay', 'OC', 'N', 'CEC', 'pH_h2o', 'ph_CaCl2']


def parse_task(task_name):
    """'Soil_Cyprus-CaCO3'  →  ('Cyprus', 'CaCO3')"""
    # strip leading 'Soil_'
    rest = task_name.replace('Soil_', '', 1)
    region, prop = rest.split('-', 1)
    return region, prop


def load_task_r2(df):
    """Return dict {task_name: ndarray shape (196,)} with mean R² over repeats."""
    task_r2 = {}
    for task, grp in df.groupby('task'):
        # mean over repeats for each patch
        r2_mean = grp.groupby('patch_idx')['r2'].mean().sort_index().values
        task_r2[task] = r2_mean
    return task_r2


def find_best_linprobe_dir(region_gadf2d_dir):
    """Scan linprobe_* subdirs and return the one with highest mean R²."""
    best_dir, best_r2 = None, -np.inf
    for name in sorted(os.listdir(region_gadf2d_dir)):
        if not name.startswith('linprobe'):
            continue
        rpr = os.path.join(region_gadf2d_dir, name, 'results_per_repeat.csv')
        rpt = os.path.join(region_gadf2d_dir, name, 'results_per_task.csv')
        if not (os.path.exists(rpr) and os.path.exists(rpt)):
            continue
        try:
            df = pd.read_csv(rpr)
            mean_r2 = float(df[df['repeat'] == 'MEAN']['r2'].values[0])
            if mean_r2 > best_r2:
                best_r2, best_dir = mean_r2, os.path.join(region_gadf2d_dir, name)
        except Exception:
            continue
    return best_dir, best_r2


def load_linprobe_task_r2(results_per_task_csv):
    """Return {task_name: mean_r2} averaged over repeats from linprobe results."""
    df = pd.read_csv(results_per_task_csv)
    df = df[pd.to_numeric(df['repeat'], errors='coerce').notna()]
    return df.groupby('task')['r2'].mean().to_dict()


def make_property_figure(prop, task_r2, save_dir, region_order=None,
                         individual_scale=False):
    """One figure per property: each column is a region."""
    tasks_for_prop = {
        task: r2 for task, r2 in task_r2.items()
        if parse_task(task)[1] == prop
    }
    if not tasks_for_prop:
        return

    all_regions = sorted(set(parse_task(t)[0] for t in tasks_for_prop))
    if region_order:
        regions = [r for r in region_order if r in all_regions]
        regions += [r for r in all_regions if r not in region_order]
    else:
        regions = all_regions
    n_cols = len(regions)

    # shared color scale (used when individual_scale=False)
    all_vals = np.concatenate([v for v in tasks_for_prop.values()])
    shared_vmin = float(np.nanpercentile(all_vals, 2))
    shared_vmax = float(np.nanpercentile(all_vals, 98))
    cmap = 'YlGn'

    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.5),
                             squeeze=False)
    scale_label = '(escala individual)' if individual_scale else '(escala compartida)'
    fig.suptitle(f'Per-patch R²  —  {prop}  {scale_label}',
                 fontsize=13, fontweight='bold')

    for col_idx, region in enumerate(regions):
        ax = axes[0][col_idx]
        task_name = f'Soil_{region}-{prop}'
        if task_name not in tasks_for_prop:
            ax.axis('off')
            ax.set_title(f'{region}\n(no data)', fontsize=9)
            continue

        r2 = tasks_for_prop[task_name].reshape(GRID_SIZE, GRID_SIZE)
        if individual_scale:
            vals = r2.flatten()
            vmin = float(np.nanpercentile(vals, 2))
            vmax = float(np.nanpercentile(vals, 98))
        else:
            vmin, vmax = shared_vmin, shared_vmax
        im = ax.imshow(r2, cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation='nearest', aspect='equal')
        ax.set_title(region, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # annotate best patch (skip if all-NaN)
        valid = tasks_for_prop[task_name]
        if np.any(np.isfinite(valid)):
            best_flat = int(np.nanargmax(valid))
            br, bc = best_flat // GRID_SIZE, best_flat % GRID_SIZE
            ax.add_patch(plt.Rectangle(
                (bc - 0.5, br - 0.5), 1, 1,
                fill=False, edgecolor='blue', linewidth=1.5))

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = os.path.join(save_dir, f'heatmap_{prop}.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f'  Saved: {out}')


def make_region_figure(region, task_r2, save_dir, individual_scale=False):
    """One figure per region: each column is a property."""
    tasks_for_region = {
        task: r2 for task, r2 in task_r2.items()
        if parse_task(task)[0] == region
    }
    if not tasks_for_region:
        return

    props = [p for p in PROPERTY_ORDER if any(
        parse_task(t)[1] == p for t in tasks_for_region)]
    n_cols = len(props)

    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.5),
                             squeeze=False)
    fig.suptitle(f'Per-patch R²  —  {region}', fontsize=13, fontweight='bold')

    for col_idx, prop in enumerate(props):
        ax = axes[0][col_idx]
        task_name = f'Soil_{region}-{prop}'
        if task_name not in tasks_for_region:
            ax.axis('off')
            ax.set_title(prop, fontsize=9)
            continue

        r2 = tasks_for_region[task_name].reshape(GRID_SIZE, GRID_SIZE)
        cmap = 'YlGn'
        if individual_scale:
            vmin_t = float(np.nanpercentile(r2, 2))
            vmax_t = float(np.nanpercentile(r2, 98))
        else:
            all_region = np.concatenate(list(tasks_for_region.values()))
            vmin_t = float(np.nanpercentile(all_region, 2))
            vmax_t = float(np.nanpercentile(all_region, 98))

        im = ax.imshow(r2, cmap=cmap, vmin=vmin_t, vmax=vmax_t,
                       interpolation='nearest', aspect='equal')
        ax.set_title(prop, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

        valid = tasks_for_region[task_name]
        if np.any(np.isfinite(valid)):
            best_flat = int(np.nanargmax(valid))
            br, bc = best_flat // GRID_SIZE, best_flat % GRID_SIZE
            ax.add_patch(plt.Rectangle(
                (bc - 0.5, br - 0.5), 1, 1,
                fill=False, edgecolor='blue', linewidth=1.5))

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    out = os.path.join(save_dir, f'heatmap_{region}.png')
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f'  Saved: {out}')


def make_overview_figure(task_r2, save_dir, region_order=None,
                         individual_scale=False):
    """Big grid: rows = properties, columns = regions. Shared scale per row."""
    all_regions = sorted(set(parse_task(t)[0] for t in task_r2))
    if region_order:
        regions = [r for r in region_order if r in all_regions]
        regions += [r for r in all_regions if r not in region_order]
    else:
        regions = all_regions

    props = [p for p in PROPERTY_ORDER if any(
        parse_task(t)[1] == p for t in task_r2)]

    fig, axes = plt.subplots(
        len(props), len(regions),
        figsize=(2.6 * len(regions), 2.6 * len(props)),
        squeeze=False)

    fig.suptitle('Per-patch R² heatmaps (14×14 patches)\nRows: property  |  Columns: region',
                 fontsize=12, fontweight='bold')

    for row_idx, prop in enumerate(props):
        row_vals = []
        for region in regions:
            t = f'Soil_{region}-{prop}'
            if t in task_r2:
                row_vals.append(task_r2[t])
        if not row_vals:
            for col_idx in range(len(regions)):
                axes[row_idx][col_idx].axis('off')
            continue

        all_row = np.concatenate(row_vals)
        shared_vmin = float(np.nanpercentile(all_row, 2))
        shared_vmax = float(np.nanpercentile(all_row, 98))
        cmap = 'YlGn'

        for col_idx, region in enumerate(regions):
            ax = axes[row_idx][col_idx]
            t = f'Soil_{region}-{prop}'

            if row_idx == 0:
                ax.set_title(region, fontsize=9, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(prop, fontsize=9, fontweight='bold')

            if t not in task_r2:
                ax.set_facecolor('#eeeeee')
                ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            r2_grid = task_r2[t].reshape(GRID_SIZE, GRID_SIZE)
            if individual_scale:
                vals = r2_grid.flatten()
                vmin = float(np.nanpercentile(vals, 2))
                vmax = float(np.nanpercentile(vals, 98))
            else:
                vmin, vmax = shared_vmin, shared_vmax
            im = ax.imshow(r2_grid, cmap=cmap, vmin=vmin, vmax=vmax,
                           interpolation='nearest', aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])

            if np.any(np.isfinite(task_r2[t])):
                best_flat = int(np.nanargmax(task_r2[t]))
                br, bc = best_flat // GRID_SIZE, best_flat % GRID_SIZE
                ax.add_patch(plt.Rectangle(
                    (bc - 0.5, br - 0.5), 1, 1,
                    fill=False, edgecolor='blue', linewidth=1.2))

            if col_idx == len(regions) - 1:
                plt.colorbar(im, ax=ax, fraction=0.07, pad=0.04)

    fig.tight_layout()
    out = os.path.join(save_dir, 'heatmap_overview.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


# ============================================================================
# Histogram figures
# ============================================================================

def make_property_histogram(prop, task_r2, save_dir, linprobe_r2=None,
                            region_order=None):
    """One figure per property: distribution of R² across 196 patches per region."""
    tasks_for_prop = {
        task: r2 for task, r2 in task_r2.items()
        if parse_task(task)[1] == prop
    }
    if not tasks_for_prop:
        return

    all_regions = sorted(set(parse_task(t)[0] for t in tasks_for_prop))
    if region_order:
        regions = [r for r in region_order if r in all_regions]
        regions += [r for r in all_regions if r not in region_order]
    else:
        regions = all_regions
    n_cols = len(regions)

    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.5),
                             squeeze=False)
    fig.suptitle(f'R² distribution across patches  —  {prop}',
                 fontsize=13, fontweight='bold')

    all_vals = np.concatenate([v for v in tasks_for_prop.values()])
    xmin = float(np.nanpercentile(all_vals, 1))
    xmax = float(np.nanpercentile(all_vals, 99))
    # extend range to include linprobe values if needed
    if linprobe_r2:
        for t in tasks_for_prop:
            if t in linprobe_r2:
                xmax = max(xmax, linprobe_r2[t] + 0.05)
    bins = np.linspace(xmin, xmax, 25)

    for col_idx, region in enumerate(regions):
        ax = axes[0][col_idx]
        task_name = f'Soil_{region}-{prop}'
        ax.set_title(region, fontsize=10)

        if task_name not in tasks_for_prop:
            ax.axis('off')
            continue

        r2 = tasks_for_prop[task_name]
        mean_r2 = float(np.nanmean(r2))
        best_r2 = float(np.nanmax(r2))

        ax.hist(r2, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(best_r2, color='blue', linewidth=1.5, linestyle=':',
                   label=f'best={best_r2:.2f}')
        if linprobe_r2 and task_name in linprobe_r2:
            lp = linprobe_r2[task_name]
            ax.axvline(lp, color='green', linewidth=2, linestyle='-',
                       label=f'avg_pool={lp:.2f}')
        ax.axvline(0, color='red', linewidth=0.8, linestyle='-', alpha=0.5)
        ax.set_xlabel('R²', fontsize=9)
        if col_idx == 0:
            ax.set_ylabel('Patches', fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlim(xmin, xmax)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(save_dir, f'hist_{prop}.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')


def make_region_histogram(region, task_r2, save_dir, linprobe_r2=None):
    """One figure per region: distribution of R² across 196 patches per property."""
    tasks_for_region = {
        task: r2 for task, r2 in task_r2.items()
        if parse_task(task)[0] == region
    }
    if not tasks_for_region:
        return

    props = [p for p in PROPERTY_ORDER if any(
        parse_task(t)[1] == p for t in tasks_for_region)]
    n_cols = len(props)

    fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.5),
                             squeeze=False)
    fig.suptitle(f'R² distribution across patches  —  {region}',
                 fontsize=13, fontweight='bold')

    for col_idx, prop in enumerate(props):
        ax = axes[0][col_idx]
        task_name = f'Soil_{region}-{prop}'
        ax.set_title(prop, fontsize=9)

        if task_name not in tasks_for_region:
            ax.axis('off')
            continue

        r2 = tasks_for_region[task_name]
        mean_r2 = float(np.nanmean(r2))
        best_r2 = float(np.nanmax(r2))
        xmin = float(np.nanpercentile(r2, 1))
        xmax = float(np.nanpercentile(r2, 99))
        if linprobe_r2 and task_name in linprobe_r2:
            xmax = max(xmax, linprobe_r2[task_name] + 0.05)
        bins = np.linspace(xmin, xmax, 25)

        ax.hist(r2, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(best_r2, color='blue', linewidth=1.5, linestyle=':',
                   label=f'best={best_r2:.2f}')
        if linprobe_r2 and task_name in linprobe_r2:
            lp = linprobe_r2[task_name]
            ax.axvline(lp, color='green', linewidth=2, linestyle='-',
                       label=f'avg_pool={lp:.2f}')
        ax.axvline(0, color='red', linewidth=0.8, linestyle='-', alpha=0.5)
        ax.set_xlabel('R²', fontsize=9)
        if col_idx == 0:
            ax.set_ylabel('Patches', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(save_dir, f'hist_{region}.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out}')


def make_overview_histogram(task_r2, save_dir, linprobe_r2=None,
                            region_order=None):
    """Big grid: rows = properties, columns = regions."""
    all_regions = sorted(set(parse_task(t)[0] for t in task_r2))
    if region_order:
        regions = [r for r in region_order if r in all_regions]
        regions += [r for r in all_regions if r not in region_order]
    else:
        regions = all_regions

    props = [p for p in PROPERTY_ORDER if any(
        parse_task(t)[1] == p for t in task_r2)]

    fig, axes = plt.subplots(
        len(props), len(regions),
        figsize=(2.8 * len(regions), 2.6 * len(props)),
        squeeze=False)

    fig.suptitle('R² distribution across 196 patches\nRows: property  |  Columns: region',
                 fontsize=12, fontweight='bold')

    for row_idx, prop in enumerate(props):
        row_vals = []
        for region in regions:
            t = f'Soil_{region}-{prop}'
            if t in task_r2:
                row_vals.append(task_r2[t])
        if not row_vals:
            for col_idx in range(len(regions)):
                axes[row_idx][col_idx].axis('off')
            continue

        all_row = np.concatenate(row_vals)
        xmin = float(np.nanpercentile(all_row, 1))
        xmax = float(np.nanpercentile(all_row, 99))
        if linprobe_r2:
            for region in regions:
                t = f'Soil_{region}-{prop}'
                if t in linprobe_r2:
                    xmax = max(xmax, linprobe_r2[t] + 0.05)
        bins = np.linspace(xmin, xmax, 20)

        for col_idx, region in enumerate(regions):
            ax = axes[row_idx][col_idx]
            t = f'Soil_{region}-{prop}'

            if row_idx == 0:
                ax.set_title(region, fontsize=9, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(prop, fontsize=9, fontweight='bold')

            if t not in task_r2:
                ax.set_facecolor('#eeeeee')
                ax.text(0.5, 0.5, 'n/a', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            r2 = task_r2[t]
            best_r2 = float(np.nanmax(r2))

            ax.hist(r2, bins=bins, edgecolor='none', alpha=0.75,
                    color='steelblue')
            ax.axvline(best_r2, color='blue', linewidth=1.2, linestyle=':')
            if linprobe_r2 and t in linprobe_r2:
                lp = linprobe_r2[t]
                ax.axvline(lp, color='green', linewidth=1.8, linestyle='-')
            ax.axvline(0, color='red', linewidth=0.7, linestyle='-', alpha=0.5)
            ax.set_xlim(xmin, xmax)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)

            lp_str = ''
            if linprobe_r2 and t in linprobe_r2:
                lp_str = f'\navg_pool={linprobe_r2[t]:.2f}'
            ax.text(0.97, 0.95,
                    f'best={best_r2:.2f}{lp_str}',
                    transform=ax.transAxes, fontsize=6,
                    ha='right', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue',  lw=1.5, linestyle=':',  label='best patch'),
        Line2D([0], [0], color='green', lw=2.0, linestyle='-',  label='avg pool'),
        Line2D([0], [0], color='red',   lw=1.0, linestyle='-',  label='R²=0'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.02, 1, 1])
    out = os.path.join(save_dir, 'hist_overview.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {out}')


def main(args):
    csv_path = os.path.join(args.results_dir, 'patch_probe_per_task.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f'Not found: {csv_path}')

    os.makedirs(args.save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} rows, {df["task"].nunique()} tasks')

    task_r2 = load_task_r2(df)

    # Derive regions and properties from data
    regions = sorted(set(parse_task(t)[0] for t in task_r2))
    props   = [p for p in PROPERTY_ORDER
               if any(parse_task(t)[1] == p for t in task_r2)]
    print(f'Regions ({len(regions)}): {regions}')
    print(f'Properties ({len(props)}): {props}')

    # -- Load best linprobe reference (optional)
    linprobe_r2 = None
    if args.linprobe_dir and os.path.isdir(args.linprobe_dir):
        best_dir, best_mean = find_best_linprobe_dir(args.linprobe_dir)
        if best_dir:
            rpt = os.path.join(best_dir, 'results_per_task.csv')
            linprobe_r2 = load_linprobe_task_r2(rpt)
            print(f'Linprobe reference: {os.path.basename(best_dir)} '
                  f'(mean R²={best_mean:.4f}, {len(linprobe_r2)} tasks)')
        else:
            print('INFO: no linprobe_* subdirs found, skipping reference line')
    else:
        print('INFO: --linprobe_dir not provided or not found, skipping reference line')

    indiv = args.individual_scale
    if indiv:
        print('INFO: using individual color scale per cell')

    print('\n--- Per-property figures (each column = region) ---')
    for prop in props:
        make_property_figure(prop, task_r2, args.save_dir,
                             region_order=regions,
                             individual_scale=indiv)

    print('\n--- Per-region figures (each column = property) ---')
    for region in regions:
        make_region_figure(region, task_r2, args.save_dir,
                           individual_scale=indiv)

    print('\n--- Overview heatmap (all tasks) ---')
    make_overview_figure(task_r2, args.save_dir, region_order=regions,
                         individual_scale=indiv)

    print('\n--- Per-property histograms ---')
    for prop in props:
        make_property_histogram(prop, task_r2, args.save_dir, linprobe_r2,
                                region_order=regions)

    print('\n--- Per-region histograms ---')
    for region in regions:
        make_region_histogram(region, task_r2, args.save_dir, linprobe_r2)

    print('\n--- Overview histogram ---')
    make_overview_histogram(task_r2, args.save_dir, linprobe_r2,
                            region_order=regions)

    print('\nDone.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--results_dir',
                   default='results/per_patch_probe_joint_ep700_train/')
    p.add_argument('--save_dir',
                   default='results/per_patch_probe_joint_ep700_train/per_task_heatmaps/')
    p.add_argument('--linprobe_dir', default=None,
                   help='Dir with linprobe_* subdirs for avg pool reference line (optional)')
    p.add_argument('--individual_scale', action='store_true',
                   help='Use individual color scale per cell instead of shared scale per property')
    main(p.parse_args())
