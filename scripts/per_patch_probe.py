"""Per-patch linear probing analysis for I-JEPA 2D (GADF).

Trains an independent linear probe (LayerNorm + Linear) for each of the 196
patch positions in the ViT encoder, across all downstream tasks.  This reveals
which spatial positions in the GADF image carry the most predictive information.

The training conditions replicate exactly those of main_finetune_gadf2d.py
with --linear_probing:
  - Head: LayerNorm(D) + Linear(D, 1)
  - Optimizer: AdamW (betas=0.9/0.95, wd=1e-5)
  - Scheduler: ReduceLROnPlateau (factor=0.3, patience=10, threshold=1e-4)
  - Early stopping: patience=30
  - Max epochs: 1000
  - Batch size: 16
  - Final evaluation on the FULL query set (not just k_qry samples)
  - n_repeats: 3

Outputs:
  - patch_probe_results.csv    (196 rows: mean/std R² and RMSE per patch)
  - patch_probe_per_task.csv   (per-task, per-repeat detail)
  - summary.json               (best/worst patch, global avg pool baseline, etc.)
  - patch_r2_heatmap.png       (14x14 heatmap of mean R²)
  - patch_r2_histogram.png     (distribution of R² across patches)

Usage:
  python scripts/per_patch_probe.py \
      --data_path data/Soil_NIR_AGG_mixed_gadf2d \
      --region_tasks --split test \
      --model_weight logs/gadf_soil_nir/gadf_jepa-latest.pth.tar \
      --save_path results/per_patch_probe/ \
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score

# -- Project root on path so we can import from main_finetune_gadf2d and src
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from main_finetune_gadf2d import load_ijepa2d_encoder, SimpleTask2D


# ============================================================================
# Feature extraction
# ============================================================================

def extract_features_batched(encoder, images, device, batch_size=32):
    """Extract encoder features in batches. Returns [N, n_patches, D] on CPU."""
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        with torch.no_grad():
            feats = encoder(batch, masks=None)
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


# ============================================================================
# Per-patch probing (same conditions as main_finetune_gadf2d --linear_probing)
# ============================================================================

def train_probe(supp_feats, supp_y, val_feats, val_y,
                embed_dim, device, lr, wd, epochs, patience,
                scheduler_patience, batch_size):
    """Train a LayerNorm+Linear probe. Returns best head (on device).

    Replicates the exact training loop of main_finetune_gadf2d.py with
    --linear_probing: AdamW, ReduceLROnPlateau, mini-batch training,
    early stopping.
    """
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

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    val_loss = float('inf')

    for _ in range(epochs):
        # Train
        head.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = head(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        # Scheduler steps with previous epoch's val_loss (matches original)
        scheduler.step(val_loss)

        # Validate
        head.eval()
        with torch.no_grad():
            val_pred = head(val_feats)
            val_loss = loss_fn(val_pred, val_y).item()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    head.load_state_dict(best_state)
    return head


def evaluate_probe(head, feats, y):
    """Evaluate a trained probe on features. Returns (r2, rmse)."""
    head.eval()
    with torch.no_grad():
        pred = head(feats)
        mse = nn.MSELoss()(pred, y).item()

    y_true = y.cpu().numpy().flatten()
    y_pred = pred.cpu().numpy().flatten()

    if np.std(y_true) < 1e-8:
        return float('nan'), float('nan')
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    return r2, rmse


# ============================================================================
# Visualization
# ============================================================================

def make_heatmaps(r2_per_patch, r2_std_per_patch, save_path, n_tasks,
                  grid_size=14):
    """Generate 14x14 heatmap of mean R² per patch position."""
    r2_grid = r2_per_patch.reshape(grid_size, grid_size)
    std_grid = r2_std_per_patch.reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mean R²
    im0 = axes[0].imshow(r2_grid, cmap='RdYlGn', interpolation='nearest')
    axes[0].set_title(f'Mean R² per patch ({n_tasks} tasks)')
    axes[0].set_xlabel('Column (patch x)')
    axes[0].set_ylabel('Row (patch y)')
    plt.colorbar(im0, ax=axes[0])
    for i in range(grid_size):
        for j in range(grid_size):
            axes[0].text(j, i, f'{r2_grid[i, j]:.2f}',
                         ha='center', va='center', fontsize=4.5,
                         color='black')

    # Std R²
    im1 = axes[1].imshow(std_grid, cmap='Reds', interpolation='nearest')
    axes[1].set_title(f'R² std across tasks')
    axes[1].set_xlabel('Column (patch x)')
    axes[1].set_ylabel('Row (patch y)')
    plt.colorbar(im1, ax=axes[1])

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'patch_r2_heatmap.png'), dpi=200)
    plt.close(fig)


def make_histogram(r2_per_patch, global_avg_r2, save_path):
    """Distribution of R² across 196 patches with global avg pool reference."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(r2_per_patch, bins=30, edgecolor='black', alpha=0.7,
            label='Per-patch R²')
    ax.axvline(global_avg_r2, color='red', linewidth=2, linestyle='--',
               label=f'Global avg pool R² = {global_avg_r2:.3f}')
    ax.set_xlabel('R²')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of per-patch R² (196 patches)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'patch_r2_histogram.png'), dpi=150)
    plt.close(fig)


# ============================================================================
# Argument parser
# ============================================================================

def get_args_parser():
    p = argparse.ArgumentParser('Per-patch linear probing analysis',
                                add_help=False)

    # Data
    p.add_argument('--data_path',
                   default='data/Soil_NIR_AGG_mixed_gadf2d', type=str)
    p.add_argument('--split', default='test', type=str)
    p.add_argument('--device', default='cuda', type=str)
    p.add_argument('--region_tasks', action='store_true')

    # Model / checkpoint
    p.add_argument('--model_weight', default='', type=str,
                   help='Path to I-JEPA 2D pretraining checkpoint')
    p.add_argument('--model_name', default='vit_base', type=str)
    p.add_argument('--patch_size', default=16, type=int)
    p.add_argument('--crop_size', default=224, type=int)

    # GADF
    p.add_argument('--gadf_image_size', default=224, type=int)
    p.add_argument('--gadf_norm_stats', default='data/gadf_norm_stats.json',
                   type=str)
    p.add_argument('--encode_diagonal', action='store_true')
    p.add_argument('--gadf_global_stats',
                   default='data/gadf_paa_global_stats.json', type=str)

    # Few-shot
    p.add_argument('--k_spt', type=int, default=25)
    p.add_argument('--k_qry', type=int, default=25)
    p.add_argument('--no_scale_y', action='store_true')

    # Training (defaults match main_finetune_gadf2d.py --linear_probing)
    p.add_argument('--epoch', type=int, default=1000)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=30,
                   help='Early stopping patience')
    p.add_argument('--scheduler_patience', type=int, default=10,
                   help='ReduceLROnPlateau patience')
    p.add_argument('--n_repeats', type=int, default=3)

    # Output
    p.add_argument('--save_path', default='results/per_patch_probe/', type=str)

    # Pilot / debug
    p.add_argument('--max_tasks', type=int, default=None,
                   help='Limit number of tasks (None = use all)')

    return p


# ============================================================================
# Main
# ============================================================================

def main(args):
    print(f'Per-patch linear probing analysis')
    print(f'{args}'.replace(', ', ',\n'))

    device = torch.device(args.device)
    os.makedirs(args.save_path, exist_ok=True)

    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # -- Load encoder
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

    encoder = encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    n_patches = (args.crop_size // args.patch_size) ** 2
    grid_size = args.crop_size // args.patch_size
    print(f'Patches: {n_patches} ({grid_size}x{grid_size}), embed_dim: {embed_dim}')

    # -- Norm stats
    if os.path.exists(args.gadf_norm_stats):
        from src.gadf_utils import load_norm_stats
        norm_mean, norm_std = load_norm_stats(args.gadf_norm_stats,
                                              use_diagonal=args.encode_diagonal)
        norm_stats = (norm_mean[0], norm_std[0])
        variant = 'with_diagonal' if args.encode_diagonal else 'no_diagonal'
        print(f'Norm stats [{variant}]: mean={norm_stats[0]:.6f}, '
              f'std={norm_stats[1]:.6f}')
    else:
        norm_stats = (-0.0000, 0.5922)
        print(f'WARNING: {args.gadf_norm_stats} not found, using defaults')

    scale_y = not args.no_scale_y

    global_stats = None
    if args.encode_diagonal:
        from src.gadf_utils import load_global_stats
        global_stats = load_global_stats(args.gadf_global_stats)
        print(f'Diagonal encoding ON (min={global_stats[0]:.4f}, '
              f'max={global_stats[1]:.4f})')

    # -- Load tasks
    if not args.region_tasks:
        print('ERROR: currently only --region_tasks mode is supported')
        sys.exit(1)

    splits_file = os.path.join(args.data_path, 'splits.csv')
    if os.path.exists(splits_file):
        splits_df = pd.read_csv(splits_file, index_col=0)
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

    print(f'\nLoaded {len(tasks)} tasks')
    if len(tasks) == 0:
        print('ERROR: no tasks loaded')
        sys.exit(1)
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
        print(f'max_tasks={args.max_tasks}: using {len(tasks)} tasks')

    n_repeats = args.n_repeats
    print(f'Running {n_repeats} repeat(s) per task')

    # -- Per-task probing
    # Collect results: [n_tasks * n_repeats, n_patches]
    all_r2 = []
    all_rmse = []
    global_r2_list = []
    global_rmse_list = []
    per_task_rows = []

    t0 = time.time()

    for ti, task in enumerate(tasks):
        print(f'\n{"="*60}')
        print(f'Task {ti+1}/{len(tasks)}: {task.name}')
        print(f'{"="*60}')

        # -- Extract features for FULL query set (for final evaluation)
        full_query_feats = extract_features_batched(
            encoder, task.query_x, device)                    # [N_q_full, 196, D] CPU
        full_query_y = task.query_y                           # [N_q_full, 1] CPU
        full_n_query = full_query_feats.shape[0]
        print(f'  Full query set: {full_n_query} samples')

        for repeat in range(1, n_repeats + 1):
            torch.manual_seed(42 + repeat)
            np.random.seed(42 + repeat)
            print(f'\n  --- Repeat {repeat}/{n_repeats} ---')

            # -- Sample support/query for training/validation
            data = task.sample_fixed(args.k_spt, args.k_qry)
            support_x = data['support_features']     # [N_s, 1, H, W]
            support_y = data['support_targets']       # [N_s, 1]
            val_query_x = data['query_features']      # [N_val, 1, H, W]
            val_query_y = data['query_targets']        # [N_val, 1]

            print(f'  Support: {support_x.shape[0]}, '
                  f'Val query: {val_query_x.shape[0]}, '
                  f'Full query: {full_n_query}')

            # -- Extract features for sampled support & val query
            with torch.no_grad():
                support_feats = encoder(
                    support_x.to(device), masks=None).cpu()   # [N_s, 196, D]
                val_feats = encoder(
                    val_query_x.to(device), masks=None).cpu() # [N_val, 196, D]

            # -- Per-patch probing
            r2_list = []
            rmse_list = []

            for i in range(n_patches):
                # Move only the relevant patch slice to GPU
                sf = support_feats[:, i, :].to(device)
                sy = support_y.to(device)
                vf = val_feats[:, i, :].to(device)
                vy = val_query_y.to(device)

                head = train_probe(
                    sf, sy, vf, vy,
                    embed_dim=embed_dim, device=device,
                    lr=args.lr, wd=args.wd,
                    epochs=args.epoch, patience=args.patience,
                    scheduler_patience=args.scheduler_patience,
                    batch_size=args.batch_size)

                # Evaluate on FULL query set
                fqf = full_query_feats[:, i, :].to(device)
                fqy = full_query_y.to(device)
                r2, rmse = evaluate_probe(head, fqf, fqy)

                r2_list.append(r2)
                rmse_list.append(rmse)

                # Free GPU memory
                del head, sf, sy, vf, vy, fqf, fqy

            # -- Global avg pool baseline (same training conditions)
            supp_avg = support_feats.mean(dim=1).to(device)   # [N_s, D]
            val_avg = val_feats.mean(dim=1).to(device)         # [N_val, D]

            head_avg = train_probe(
                supp_avg, support_y.to(device),
                val_avg, val_query_y.to(device),
                embed_dim=embed_dim, device=device,
                lr=args.lr, wd=args.wd,
                epochs=args.epoch, patience=args.patience,
                scheduler_patience=args.scheduler_patience,
                batch_size=args.batch_size)

            fq_avg = full_query_feats.mean(dim=1).to(device)
            g_r2, g_rmse = evaluate_probe(
                head_avg, fq_avg, full_query_y.to(device))
            del head_avg, supp_avg, val_avg, fq_avg

            all_r2.append(r2_list)
            all_rmse.append(rmse_list)
            global_r2_list.append(g_r2)
            global_rmse_list.append(g_rmse)

            # Per-task detail rows
            for i in range(n_patches):
                per_task_rows.append({
                    'task': task.name,
                    'repeat': repeat,
                    'patch_idx': i,
                    'row': i // grid_size,
                    'col': i % grid_size,
                    'r2': r2_list[i],
                    'rmse': rmse_list[i],
                })

            print(f'  Global avg pool: R²={g_r2:.4f}, RMSE={g_rmse:.4f}')
            valid_r2 = [v for v in r2_list if not np.isnan(v)]
            if valid_r2:
                print(f'  Per-patch R²: min={min(valid_r2):.4f}, '
                      f'max={max(valid_r2):.4f}, mean={np.mean(valid_r2):.4f}')

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed:.1f}s')

    # -- Aggregate: first mean across repeats per task, then across tasks
    all_r2 = np.array(all_r2)       # [n_tasks * n_repeats, 196]
    all_rmse = np.array(all_rmse)

    # Reshape to [n_tasks, n_repeats, 196], mean over repeats
    n_t = len(tasks)
    r2_by_task = all_r2.reshape(n_t, n_repeats, n_patches)
    rmse_by_task = all_rmse.reshape(n_t, n_repeats, n_patches)

    r2_task_mean = np.nanmean(r2_by_task, axis=1)     # [n_tasks, 196]
    rmse_task_mean = np.nanmean(rmse_by_task, axis=1)

    # Then mean/std across tasks
    r2_mean = np.nanmean(r2_task_mean, axis=0)         # [196]
    r2_std = np.nanstd(r2_task_mean, axis=0)
    rmse_mean = np.nanmean(rmse_task_mean, axis=0)
    rmse_std = np.nanstd(rmse_task_mean, axis=0)

    global_r2_arr = np.array(global_r2_list).reshape(n_t, n_repeats)
    global_r2_per_task = np.nanmean(global_r2_arr, axis=1)
    global_r2_mean = np.nanmean(global_r2_per_task)
    global_r2_std = np.nanstd(global_r2_per_task)

    global_rmse_arr = np.array(global_rmse_list).reshape(n_t, n_repeats)
    global_rmse_per_task = np.nanmean(global_rmse_arr, axis=1)
    global_rmse_mean = np.nanmean(global_rmse_per_task)

    # -- Save aggregated results
    results_df = pd.DataFrame({
        'patch_idx': range(n_patches),
        'row': [i // grid_size for i in range(n_patches)],
        'col': [i % grid_size for i in range(n_patches)],
        'r2_mean': r2_mean,
        'r2_std': r2_std,
        'rmse_mean': rmse_mean,
        'rmse_std': rmse_std,
    })
    results_df.to_csv(os.path.join(args.save_path, 'patch_probe_results.csv'),
                      index=False)

    # -- Save per-task detail
    per_task_df = pd.DataFrame(per_task_rows)
    per_task_df.to_csv(os.path.join(args.save_path, 'patch_probe_per_task.csv'),
                       index=False)

    # -- Symmetry check (GADF is symmetric → heatmap should be ~symmetric)
    r2_grid = r2_mean.reshape(grid_size, grid_size)
    valid_mask = ~np.isnan(r2_grid) & ~np.isnan(r2_grid.T)
    if valid_mask.sum() > 0:
        symmetry_corr = np.corrcoef(
            r2_grid[valid_mask].flatten(),
            r2_grid.T[valid_mask].flatten()
        )[0, 1]
    else:
        symmetry_corr = float('nan')

    # -- Summary JSON
    best_idx = int(np.nanargmax(r2_mean))
    worst_idx = int(np.nanargmin(r2_mean))

    summary = {
        'n_tasks': len(tasks),
        'n_repeats': n_repeats,
        'n_patches': n_patches,
        'grid_size': grid_size,
        'global_avg_pool_r2': float(global_r2_mean),
        'global_avg_pool_r2_std': float(global_r2_std),
        'global_avg_pool_rmse': float(global_rmse_mean),
        'best_patch_idx': best_idx,
        'best_patch_row': best_idx // grid_size,
        'best_patch_col': best_idx % grid_size,
        'best_patch_r2': float(r2_mean[best_idx]),
        'worst_patch_idx': worst_idx,
        'worst_patch_row': worst_idx // grid_size,
        'worst_patch_col': worst_idx % grid_size,
        'worst_patch_r2': float(r2_mean[worst_idx]),
        'symmetry_correlation': float(symmetry_corr),
        'elapsed_seconds': round(elapsed, 1),
        'checkpoint': args.model_weight,
        'config': vars(args),
    }
    with open(os.path.join(args.save_path, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # -- Visualizations
    make_heatmaps(r2_mean, r2_std, args.save_path, len(tasks),
                  grid_size=grid_size)
    make_histogram(r2_mean, global_r2_mean, args.save_path)

    # -- Print summary
    print(f'\n{"="*60}')
    print('PER-PATCH PROBE SUMMARY')
    print(f'{"="*60}')
    print(f'Tasks: {len(tasks)}, Repeats: {n_repeats}')
    print(f'Global avg pool baseline: R²={global_r2_mean:.4f} +/- '
          f'{global_r2_std:.4f}, RMSE={global_rmse_mean:.4f}')
    print(f'Best patch:  idx={best_idx} (row={best_idx//grid_size}, '
          f'col={best_idx%grid_size}) R²={r2_mean[best_idx]:.4f}')
    print(f'Worst patch: idx={worst_idx} (row={worst_idx//grid_size}, '
          f'col={worst_idx%grid_size}) R²={r2_mean[worst_idx]:.4f}')
    print(f'Mean per-patch R²: {np.nanmean(r2_mean):.4f} +/- '
          f'{np.nanstd(r2_mean):.4f}')
    print(f'Symmetry correlation: {symmetry_corr:.4f}')
    print(f'\nResults saved to {args.save_path}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
