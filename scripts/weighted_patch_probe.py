"""Weighted patch pooling for downstream regression.

Uses property-level importance maps derived from training tasks
(output of analyze_patch_importance.py) to perform weighted average pooling
of ViT patch embeddings.  Multiple weighting strategies are compared against
the uniform average-pool baseline.

Weighting strategies
--------------------
  uniform   — equal weight for all 196 patches (standard avg pool)
  relu      — clamp negative R² to 0; normalise to sum=1
  softmax   — softmax(R² / T); T controlled by --softmax_temperature
  top_k     — equal weight on top-k patches only; 0 elsewhere
  rank      — rank-based weights (rank 1=worst → 196=best); normalised

Prerequisites
-------------
  1. Frozen I-JEPA checkpoint (--model_weight)
  2. importance_maps.npz from analyze_patch_importance.py (--importance_maps)
  3. Task directories with X_supp.pt, X_query.pt, y_supp.csv, y_query.csv
     (and optionally fixed_val_support_25shots.csv for deterministic splits)

Outputs (in --save_path)
------------------------
  weighted_probe_results.csv      per task × strategy × repeat
  weighted_probe_summary.csv      per strategy (mean / std across tasks)
  comparison_by_property.csv      per property × strategy
  comparison_barplot.png          grouped bar chart per property + aggregate
  strategy_heatmaps.png           14×14 weight maps per strategy × property
  config.json

Usage
-----
  python scripts/weighted_patch_probe.py \\
      --importance_maps results/importance_analysis/importance_maps.npz \\
      --model_weight    logs/gadf_soil_nir/gadf_jepa-latest.pth.tar \\
      --split test \\
      --save_path results/weighted_patch_probe/ \\
      --device cuda:0
"""

import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

# -- Project root
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from main_finetune_gadf2d import load_ijepa2d_encoder, SimpleTask2D


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE  = 14
N_PATCHES  = GRID_SIZE * GRID_SIZE  # 196
WL_START   = 401
WL_END     = 2499
PATCH_SIZE = 16

PROPERTY_ORDER = ['CaCO3', 'Clay', 'OC', 'N', 'CEC', 'pH_h2o', 'ph_CaCl2']


# ---------------------------------------------------------------------------
# Task name parsing (same as analyze_patch_importance.py)
# ---------------------------------------------------------------------------

def parse_task(task_name: str):
    rest = task_name.replace('Soil_', '', 1)
    country, prop = rest.split('-', 1)
    return country, prop


# ---------------------------------------------------------------------------
# Weighting strategies
# ---------------------------------------------------------------------------

def make_weights(r2_map: np.ndarray, strategy: str,
                 k: int = 20, temperature: float = 1.0) -> torch.Tensor:
    """Convert a [196] R² importance array into normalised weights.

    Returns a [196] float32 tensor.
    """
    r2 = torch.tensor(r2_map, dtype=torch.float32)
    r2 = torch.nan_to_num(r2, nan=0.0)

    if strategy == 'uniform':
        return torch.ones(N_PATCHES) / N_PATCHES

    if strategy == 'relu':
        w = torch.clamp(r2, min=0.0)
        s = w.sum()
        return w / s if s > 1e-8 else torch.ones(N_PATCHES) / N_PATCHES

    if strategy == 'softmax':
        # shift by mean before dividing by T for numerical stability
        w = torch.softmax((r2 - r2.mean()) / temperature, dim=0)
        return w

    if strategy == 'top_k':
        k = min(k, N_PATCHES)
        w = torch.zeros(N_PATCHES)
        topk_idx = torch.topk(r2, k).indices
        w[topk_idx] = 1.0 / k
        return w

    if strategy == 'rank':
        # argsort of argsort → ranks in [0, N_PATCHES)
        order = torch.argsort(torch.argsort(r2))
        ranks = (order + 1).float()  # [1 … 196]
        return ranks / ranks.sum()

    raise ValueError(f'Unknown strategy: {strategy}')


def weighted_pool(patches: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted mean pooling.

    patches : [N, 196, D]
    w       : [196]   (already normalised, on the same device)
    returns : [N, D]
    """
    return (patches * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1)


# ---------------------------------------------------------------------------
# Feature extraction (same as per_patch_probe.py)
# ---------------------------------------------------------------------------

def extract_features_batched(encoder, images, device, batch_size=32):
    """Returns [N, 196, D] on CPU."""
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        with torch.no_grad():
            feats = encoder(batch, masks=None)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


# ---------------------------------------------------------------------------
# Probe training / evaluation (verbatim from per_patch_probe.py)
# ---------------------------------------------------------------------------

def train_probe(supp_feats, supp_y, val_feats, val_y,
                embed_dim, device, lr, wd, epochs, patience,
                scheduler_patience, batch_size):
    head = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim, 1),
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=wd)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3,
        threshold=0.0001, patience=scheduler_patience)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(supp_feats, supp_y),
        batch_size=batch_size, shuffle=True)

    best_val_loss  = float('inf')
    patience_counter = 0
    best_state     = None
    val_loss       = float('inf')

    for _ in range(epochs):
        head.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss_fn(head(xb), yb).backward()
            optimizer.step()

        scheduler.step(val_loss)

        head.eval()
        with torch.no_grad():
            val_loss = loss_fn(head(val_feats), val_y).item()

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    head.load_state_dict(best_state)
    return head


def evaluate_probe(head, feats, y):
    """Returns (r2, rmse)."""
    head.eval()
    with torch.no_grad():
        pred = head(feats)
        mse  = nn.MSELoss()(pred, y).item()

    y_true = y.cpu().numpy().flatten()
    y_pred = pred.cpu().numpy().flatten()

    if np.std(y_true) < 1e-8:
        return float('nan'), float('nan')
    return r2_score(y_true, y_pred), float(np.sqrt(mse))


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def patch_wl_center(idx: int) -> int:
    pixel_center = idx * PATCH_SIZE + PATCH_SIZE // 2
    wl = WL_START + pixel_center * (WL_END - WL_START) / (GRID_SIZE * PATCH_SIZE - 1)
    return int(round(wl))


def plot_comparison_barplot(df_prop: pd.DataFrame, strategies: list,
                            save_path: str):
    """Grouped bar chart: strategies × properties + aggregate panel."""
    props = [p for p in PROPERTY_ORDER if p in df_prop['property'].values]
    others = sorted(set(df_prop['property'].values) - set(props))
    props = props + others
    props_with_agg = props + ['ALL']

    ncols = len(props_with_agg)
    fig, axes = plt.subplots(1, ncols, figsize=(max(ncols * 2.5, 10), 5),
                             sharey=False)
    if ncols == 1:
        axes = [axes]

    cmap = plt.get_cmap('tab10')
    colors = {s: cmap(i) for i, s in enumerate(strategies)}

    for ax, key in zip(axes, props_with_agg):
        if key == 'ALL':
            sub = df_prop.groupby('strategy')[['r2_mean']].mean().reset_index()
            sub['r2_std'] = df_prop.groupby('strategy')['r2_mean'].std().values
            title = 'ALL'
        else:
            sub = df_prop[df_prop['property'] == key]
            title = key

        x = np.arange(len(strategies))
        for xi, s in enumerate(strategies):
            row = sub[sub['strategy'] == s]
            if len(row) == 0:
                continue
            r2   = float(row['r2_mean'].values[0])
            std  = float(row['r2_std'].values[0]) if 'r2_std' in row else 0.0
            ax.bar(xi, r2, yerr=std, color=colors[s], alpha=0.8,
                   capsize=4, label=s)

        # uniform baseline as dashed line
        unif = sub[sub['strategy'] == 'uniform']
        if len(unif):
            ax.axhline(float(unif['r2_mean'].values[0]),
                       color='black', linestyle='--', linewidth=1.0,
                       alpha=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel('R²' if ax == axes[0] else '')
        ax.grid(axis='y', alpha=0.3)

    # single legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s], alpha=0.8)
               for s in strategies]
    fig.legend(handles, strategies, loc='upper right',
               fontsize=8, title='Strategy')
    fig.suptitle('Weighted pooling strategies vs. uniform baseline', fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_strategy_heatmaps(importance_maps: dict, strategies: list,
                           top_k: int, temperature: float,
                           save_path: str):
    """Weight heatmaps: rows=strategies, columns=properties."""
    props = [p for p in PROPERTY_ORDER if p in importance_maps]
    others = sorted(set(importance_maps.keys()) - set(props))
    props = props + others

    if not props:
        return

    nrows = len(strategies)
    ncols = len(props)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.0, nrows * 1.8),
                             squeeze=False)

    wl_labels = [str(patch_wl_center(i)) for i in range(GRID_SIZE)]

    for ri, s in enumerate(strategies):
        for ci, prop in enumerate(props):
            ax = axes[ri][ci]
            w = make_weights(importance_maps[prop], s,
                             k=top_k, temperature=temperature)
            grid = w.numpy().reshape(GRID_SIZE, GRID_SIZE)

            im = ax.imshow(grid, cmap='YlOrRd', interpolation='nearest',
                           aspect='equal')
            if ri == 0:
                ax.set_title(prop, fontsize=8)
            if ci == 0:
                ax.set_ylabel(s, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle('Patch weights per strategy × property\n'
                 '(brighter = higher weight)', fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def get_args_parser():
    p = argparse.ArgumentParser('Weighted patch probe', add_help=False)

    # Data
    p.add_argument('--data_path',
                   default='data/Soil_NIR_AGG_mixed_gadf2d', type=str)
    p.add_argument('--split', default='test', type=str)
    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--region_tasks', action='store_true', default=True)

    # Importance maps
    p.add_argument('--importance_maps', required=True, type=str,
                   help='Path to importance_maps.npz from analyze_patch_importance.py')

    # Model
    p.add_argument('--model_weight', default='', type=str)
    p.add_argument('--model_name', default='vit_base', type=str)
    p.add_argument('--patch_size', default=16, type=int)
    p.add_argument('--crop_size', default=224, type=int)

    # GADF
    p.add_argument('--gadf_image_size', default=224, type=int)
    p.add_argument('--gadf_norm_stats',
                   default='data/gadf_norm_stats.json', type=str)
    p.add_argument('--encode_diagonal', action='store_true')
    p.add_argument('--gadf_global_stats',
                   default='data/gadf_paa_global_stats.json', type=str)

    # Few-shot
    p.add_argument('--k_spt', type=int, default=25)
    p.add_argument('--k_qry', type=int, default=25)
    p.add_argument('--no_scale_y', action='store_true')

    # Training (same defaults as per_patch_probe.py)
    p.add_argument('--epoch', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=30)
    p.add_argument('--scheduler_patience', type=int, default=10)
    p.add_argument('--n_repeats', type=int, default=3)

    # Weighting strategies
    p.add_argument('--strategies', nargs='+',
                   default=['uniform', 'relu', 'softmax', 'top_k', 'rank'],
                   help='Space-separated list of strategies to compare')
    p.add_argument('--top_k', type=int, default=20,
                   help='Number of patches for top_k strategy')
    p.add_argument('--softmax_temperature', type=float, default=1.0,
                   help='Temperature for softmax strategy')

    # Output
    p.add_argument('--save_path', default='results/weighted_patch_probe/',
                   type=str)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print('Weighted patch probe analysis')
    print(f'{args}'.replace(', ', ',\n'))

    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # -----------------------------------------------------------------------
    # Load importance maps
    # -----------------------------------------------------------------------
    data = np.load(args.importance_maps)
    importance_maps = {k: data[k] for k in data.files}
    print(f'Loaded importance maps for: {sorted(importance_maps.keys())}')

    # -----------------------------------------------------------------------
    # Load encoder
    # -----------------------------------------------------------------------
    if args.model_weight:
        encoder, embed_dim = load_ijepa2d_encoder(
            checkpoint_path=args.model_weight,
            model_name=args.model_name,
            patch_size=args.patch_size,
            crop_size=args.crop_size,
            in_chans=1)
    else:
        import src.models.vision_transformer as vit
        encoder = vit.__dict__[args.model_name](
            img_size=[args.crop_size],
            patch_size=args.patch_size,
            in_chans=1)
        embed_dim = encoder.embed_dim
        print('Using random encoder (no pretrained weights)')

    encoder = encoder.to(device).eval()
    for param in encoder.parameters():
        param.requires_grad_(False)

    n_patches = (args.crop_size // args.patch_size) ** 2
    grid_size = args.crop_size // args.patch_size
    print(f'n_patches={n_patches}, embed_dim={embed_dim}')

    # -----------------------------------------------------------------------
    # Norm stats
    # -----------------------------------------------------------------------
    if os.path.exists(args.gadf_norm_stats):
        from src.gadf_utils import load_norm_stats
        norm_mean, norm_std = load_norm_stats(args.gadf_norm_stats,
                                              use_diagonal=args.encode_diagonal)
        norm_stats = (norm_mean[0], norm_std[0])
    else:
        norm_stats = (-0.0000, 0.5922)
        print(f'WARNING: {args.gadf_norm_stats} not found, using defaults')

    scale_y = not args.no_scale_y

    global_stats = None
    if args.encode_diagonal:
        from src.gadf_utils import load_global_stats
        global_stats = load_global_stats(args.gadf_global_stats)

    # -----------------------------------------------------------------------
    # Load tasks
    # -----------------------------------------------------------------------
    splits_file = os.path.join(args.data_path, 'splits.csv')
    if os.path.exists(splits_file):
        splits_df  = pd.read_csv(splits_file, index_col=0)
        task_names = splits_df[splits_df['split'] == args.split]['task'].tolist()
        print(f"splits.csv: {len(task_names)} tasks for split '{args.split}'")
    else:
        task_names = sorted([d for d in os.listdir(args.data_path)
                             if os.path.isdir(os.path.join(args.data_path, d))])
        print(f'No splits.csv, using all {len(task_names)} subdirectories')

    tasks = []
    for tname in task_names:
        task_dir = os.path.join(args.data_path, tname)
        if not os.path.isdir(task_dir):
            continue
        try:
            t = SimpleTask2D(
                data_path=task_dir, target_column=None,
                image_size=args.gadf_image_size,
                norm_stats=norm_stats, scale_y=scale_y,
                device=device, global_stats=global_stats)
            t.name = tname
            tasks.append(t)
        except Exception as e:
            print(f'  Skipping {tname}: {e}')

    print(f'Loaded {len(tasks)} tasks')
    if not tasks:
        print('ERROR: no tasks loaded'); sys.exit(1)

    strategies = args.strategies
    print(f'Strategies: {strategies}')

    # -----------------------------------------------------------------------
    # Per-task evaluation
    # -----------------------------------------------------------------------
    rows = []
    t0 = time.time()

    for ti, task in enumerate(tasks):
        country, prop = parse_task(task.name)
        print(f'\n[{ti+1}/{len(tasks)}] {task.name}  ({prop}, {country})')

        # Get importance weights for this property (fall back to uniform)
        if prop in importance_maps:
            r2_map = importance_maps[prop]
        else:
            print(f'  WARNING: no importance map for {prop}, using uniform')
            r2_map = np.ones(n_patches, dtype=np.float32) / n_patches

        # Pre-extract features for full query set (for final evaluation)
        full_query_feats = extract_features_batched(
            encoder, task.query_x, device)           # [N_q, 196, D]  CPU
        full_query_y     = task.query_y               # [N_q, 1]       CPU

        for repeat in range(1, args.n_repeats + 1):
            torch.manual_seed(42 + repeat)
            np.random.seed(42 + repeat)

            data_s = task.sample_fixed(args.k_spt, args.k_qry)
            supp_x     = data_s['support_features']   # [N_s, 1, H, W]
            supp_y     = data_s['support_targets']     # [N_s, 1]
            val_x      = data_s['query_features']      # [N_val, 1, H, W]
            val_y      = data_s['query_targets']        # [N_val, 1]

            # Extract patch embeddings for support and val-query
            with torch.no_grad():
                supp_patches = encoder(
                    supp_x.to(device), masks=None).cpu()  # [N_s, 196, D]
                val_patches  = encoder(
                    val_x.to(device), masks=None).cpu()   # [N_val, 196, D]

            for s in strategies:
                w = make_weights(r2_map, s,
                                 k=args.top_k,
                                 temperature=args.softmax_temperature)
                w_dev = w.to(device)

                # Pool: [N, 196, D] → [N, D]
                sf = weighted_pool(supp_patches, w_dev)
                vf = weighted_pool(val_patches,  w_dev)
                fq = weighted_pool(full_query_feats, w_dev)

                head = train_probe(
                    sf.to(device), supp_y.to(device),
                    vf.to(device), val_y.to(device),
                    embed_dim=embed_dim, device=device,
                    lr=args.lr, wd=args.wd,
                    epochs=args.epoch, patience=args.patience,
                    scheduler_patience=args.scheduler_patience,
                    batch_size=args.batch_size)

                r2, rmse = evaluate_probe(
                    head, fq.to(device), full_query_y.to(device))

                rows.append({
                    'task':     task.name,
                    'country':  country,
                    'property': prop,
                    'strategy': s,
                    'repeat':   repeat,
                    'r2':       r2,
                    'rmse':     rmse,
                })

                del head, sf, vf, fq, w_dev
                print(f'  repeat={repeat}  {s:10s}: R²={r2:.4f}  RMSE={rmse:.4f}')

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed:.1f}s')

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.save_path, 'weighted_probe_results.csv'),
              index=False)

    # Summary per strategy
    summary_rows = []
    uniform_mean = df[df['strategy'] == 'uniform']['r2'].mean()
    for s in strategies:
        sub = df[df['strategy'] == s]
        # Mean over repeats first, then over tasks
        task_means = sub.groupby('task')['r2'].mean()
        r2_m  = float(task_means.mean())
        r2_s  = float(task_means.std())
        rmse_m = float(sub.groupby('task')['rmse'].mean().mean())
        summary_rows.append({
            'strategy':             s,
            'r2_mean':              r2_m,
            'r2_std':               r2_s,
            'rmse_mean':            rmse_m,
            'n_tasks':              int(sub['task'].nunique()),
            'delta_r2_vs_uniform':  r2_m - uniform_mean,
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(args.save_path, 'weighted_probe_summary.csv'),
                      index=False)

    # Per property × strategy
    prop_rows = []
    for prop in df['property'].unique():
        sub_prop = df[df['property'] == prop]
        unif_prop = sub_prop[sub_prop['strategy'] == 'uniform']['r2'].mean()
        for s in strategies:
            sub = sub_prop[sub_prop['strategy'] == s]
            task_means = sub.groupby('task')['r2'].mean()
            r2_m  = float(task_means.mean())
            r2_s  = float(task_means.std())
            prop_rows.append({
                'property':             prop,
                'strategy':             s,
                'r2_mean':              r2_m,
                'r2_std':               r2_s,
                'n_countries':          int(sub['country'].nunique()),
                'delta_r2_vs_uniform':  r2_m - unif_prop,
            })
    df_prop = pd.DataFrame(prop_rows)
    df_prop.to_csv(os.path.join(args.save_path, 'comparison_by_property.csv'),
                   index=False)

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print(f'\n{"="*60}')
    print('SUMMARY')
    print(f'{"="*60}')
    print(df_summary.to_string(index=False, float_format='{:.4f}'.format))

    # -----------------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------------
    plot_comparison_barplot(
        df_prop, strategies,
        os.path.join(args.save_path, 'comparison_barplot.png'))
    print('\nSaved comparison_barplot.png')

    plot_strategy_heatmaps(
        importance_maps, strategies,
        args.top_k, args.softmax_temperature,
        os.path.join(args.save_path, 'strategy_heatmaps.png'))
    print('Saved strategy_heatmaps.png')

    print(f'\nAll results saved to {args.save_path}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
