# ijepa_spectra2d — ProtoNet zero-shot inference with joint pretrained checkpoint
#
# Loads target_encoder + proto_projection from a joint pretraining checkpoint and
# runs ProtoNet inference (no gradient updates) over a task split.
#
# Inference: fixed support set (from fixed_val_*) → encode → prototypes
#            full query set (via query_dataloader) → encode → distance-weighted preds
#
# Metrics and output format are identical to main_finetune_gadf2d.py so results
# are directly comparable (same normalization, same fixed_val splits, same aggregation).
#
# Usage:
#   python main_eval_joint_protonet.py \
#       --checkpoint logs/gadf_soil_nir_joint_proj/gadf_joint_proj-latest.pth.tar \
#       --config     configs/gadf_soil_nir_joint_proj.yaml \
#       --split      test \
#       --k_spt      25 \
#       --save_path  results/eval_joint_proj_test/ \
#       --device     cuda:0
#
# Requires use_projection: true in the pretraining config. For checkpoints without
# projection use:
#   python scripts/precompute_embeddings.py --checkpoint <ckpt>
#   python main_protonet_gadf2d.py --no_projection ...

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import r2_score

from main_finetune_gadf2d import SimpleTask2D, load_ijepa2d_encoder
from src.protonet import ProjectionNet, TransformerAggregator

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ---------------------------------------------------------------------------
# Projection loader
# ---------------------------------------------------------------------------

def load_proto_projection(checkpoint_path, config, encoder_embed_dim, device):
    """Reconstruct proto_projection from config and load weights from checkpoint."""
    proto_cfg = config.get('protonet', {})
    projection_type = proto_cfg.get('projection_type', 'mlp')
    hidden_dim = int(proto_cfg.get('projection_hidden_dim', 256))
    output_dim = int(proto_cfg.get('projection_output_dim', 128))
    crop_size = config['data'].get('crop_size', 224)
    patch_size = config['mask'].get('patch_size', 16)

    if projection_type == 'transformer':
        num_patches = (crop_size // patch_size) ** 2
        proj = TransformerAggregator(
            input_dim=encoder_embed_dim,
            d_model=hidden_dim,
            output_dim=output_dim,
            num_patches=num_patches,
            num_layers=int(proto_cfg.get('aggregator_num_layers', 1)),
            num_heads=int(proto_cfg.get('aggregator_num_heads', 4)),
            mlp_ratio=float(proto_cfg.get('aggregator_mlp_ratio', 2.0)),
            dropout=float(proto_cfg.get('aggregator_dropout', 0.1)),
            attn_dropout=float(proto_cfg.get('aggregator_attn_dropout', 0.1)),
        )
    else:
        proj = ProjectionNet(
            input_dim=encoder_embed_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'proto_projection' not in ckpt:
        print('ERROR: checkpoint has no proto_projection key.')
        print('This means the pretraining was run with use_projection=false.')
        print('For checkpoints without projection use:')
        print('  python scripts/precompute_embeddings.py --checkpoint <ckpt>')
        print('  python main_protonet_gadf2d.py --no_projection ...')
        sys.exit(1)

    state_dict = ckpt['proto_projection']
    cleaned = {k.replace('module.', ''): v for k, v in state_dict.items()}
    msg = proj.load_state_dict(cleaned, strict=True)
    epoch = ckpt.get('epoch', '?')
    n_params = sum(p.numel() for p in proj.parameters())
    print(f'Loaded proto_projection ({projection_type}) from epoch {epoch} '
          f'({n_params/1e3:.1f}K params)')
    if msg.missing_keys or msg.unexpected_keys:
        print(f'  load msg: {msg}')

    del ckpt
    return proj.to(device), projection_type


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    # -- load YAML config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # -- validate scope
    proto_cfg = config.get('protonet', {})
    if not proto_cfg.get('use_projection', False):
        print('ERROR: This script requires use_projection=true in the pretraining config.')
        print('The config at', args.config, 'has use_projection=false.')
        print('For checkpoints without projection use:')
        print('  python scripts/precompute_embeddings.py --checkpoint <ckpt>')
        print('  python main_protonet_gadf2d.py --no_projection ...')
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_path, exist_ok=True)

    with open(os.path.join(args.save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # -- resolve defaults from config
    data_path = args.data_path or proto_cfg.get('data_path', '')
    k_spt = args.k_spt if args.k_spt is not None else int(proto_cfg.get('k_spt', 25))
    k_qry = args.k_qry if args.k_qry is not None else int(proto_cfg.get('k_qry', 25))
    dist_temperature = (args.dist_temperature
                        if args.dist_temperature is not None
                        else float(proto_cfg.get('dist_temperature', 0.5)))
    gadf_norm_stats_path = (args.gadf_norm_stats
                            or config['data'].get('gadf_norm_stats', ''))
    gadf_global_stats_path = (args.gadf_global_stats
                              or config['data'].get('gadf_global_stats', None))
    encode_diagonal = args.encode_diagonal or (gadf_global_stats_path is not None)
    crop_size = config['data'].get('crop_size', 224)
    model_name = config['meta'].get('model_name', 'vit_base')
    patch_size = config['mask'].get('patch_size', 16)

    print(f'Checkpoint : {args.checkpoint}')
    print(f'Config     : {args.config}')
    print(f'Split      : {args.split}')
    print(f'Data path  : {data_path}')
    print(f'k_spt={k_spt}, k_qry={k_qry}, dist_T={dist_temperature}')
    print(f'Device     : {device}')

    # -- GADF norm stats
    if os.path.exists(gadf_norm_stats_path):
        from src.gadf_utils import load_norm_stats
        norm_mean, norm_std = load_norm_stats(gadf_norm_stats_path,
                                              use_diagonal=encode_diagonal)
        norm_stats = (norm_mean[0], norm_std[0])
        variant = 'with_diagonal' if encode_diagonal else 'no_diagonal'
        print(f'Norm stats [{variant}]: mean={norm_stats[0]:.6f}, std={norm_stats[1]:.6f}')
    else:
        norm_stats = (-0.0000, 0.5922)
        print(f'WARNING: {gadf_norm_stats_path} not found — using hardcoded defaults')

    # -- optional diagonal encoding
    global_stats = None
    if encode_diagonal and gadf_global_stats_path:
        from src.gadf_utils import load_global_stats
        global_stats = load_global_stats(gadf_global_stats_path)
        print(f'Diagonal encoding ON (min={global_stats[0]:.4f}, max={global_stats[1]:.4f})')

    # -- load encoder
    encoder, embed_dim = load_ijepa2d_encoder(
        checkpoint_path=args.checkpoint,
        model_name=model_name,
        patch_size=patch_size,
        crop_size=crop_size,
        in_chans=1)
    encoder = encoder.to(device)
    encoder.eval()

    # -- load projection
    proto_projection, projection_type = load_proto_projection(
        args.checkpoint, config, embed_dim, device)
    proto_projection.eval()

    # -- load tasks from split
    splits_file = os.path.join(data_path, 'splits.csv')
    if not os.path.exists(splits_file):
        print(f'ERROR: splits.csv not found at {data_path}')
        sys.exit(1)

    splits_df = pd.read_csv(splits_file, index_col=0)
    task_names = sorted(splits_df[splits_df['split'] == args.split]['task'].tolist())
    print(f'\nLoading {len(task_names)} tasks from split "{args.split}"...')

    tasks = []
    for tname in task_names:
        task_dir = os.path.join(data_path, tname)
        if not os.path.isdir(task_dir):
            continue
        try:
            t = SimpleTask2D(
                data_path=task_dir,
                image_size=crop_size,
                norm_stats=norm_stats,
                scale_y=True,
                device=device,
                global_stats=global_stats,
            )
            t.name = tname
            tasks.append(t)
        except Exception as e:
            print(f'  Skipping {tname}: {e}')

    print(f'Loaded {len(tasks)} tasks')

    # En modo fixed, verificar que existan los fixed_val_*shots.csv
    eval_mode = args.eval_mode
    if eval_mode == 'fixed':
        _sample_task = tasks[0] if tasks else None
        if _sample_task is not None:
            _supp_csv = os.path.join(_sample_task.data_path,
                                     f'fixed_val_support_{k_spt}shots.csv')
            if not os.path.exists(_supp_csv):
                available = sorted({
                    int(f.split('_')[3].replace('shots.csv', ''))
                    for f in os.listdir(_sample_task.data_path)
                    if f.startswith('fixed_val_support_') and f.endswith('shots.csv')
                })
                print(f'ERROR: fixed_val_support_{k_spt}shots.csv not found.')
                print(f'Available shot counts: {available}')
                print(f'Pass one of these as --k_spt, or use --eval_mode cv')
                sys.exit(1)
    else:
        print(f'Eval mode: K-fold CV (k_spt={k_spt}, seed=42)')

    # -- W&B
    use_wandb = HAS_WANDB and bool(args.wandb_project)
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            dir=args.save_path,
        )

    # -----------------------------------------------------------------------
    # Task loop — same structure as main_finetune_gadf2d.py
    # -----------------------------------------------------------------------
    results_per_task = []
    predictions_all_tasks = []
    n_repeats = args.n_repeats

    if eval_mode == 'cv':
        print(f'\nEval mode: K-fold CV (k_spt={k_spt}, seed=42)')
        if n_repeats != 1:
            print('  WARNING: --n_repeats ignorado en modo cv (las repeticiones son los folds)')
    else:
        print(f'\nRunning {n_repeats} repeat(s) per task')

    for task_idx, task in enumerate(tasks):
        task_name = task.name
        print(f'\n{"="*60}')
        print(f'Task {task_idx + 1}/{len(tasks)}: {task_name}')
        print(f'{"="*60}')

        if eval_mode == 'cv':
            # K-fold CV: iterar folds, recolectar métricas por fold
            fold_mses, fold_maes, fold_rmses, fold_r2s = [], [], [], []
            all_y_true, all_y_pred = [], []
            first_fold = True

            for fold in task.iter_folds(k_spt, seed=42):
                fold_idx = fold['fold_idx']
                num_folds = fold['num_folds']
                support_x = fold['support_features']
                support_y = fold['support_targets']
                query_x_fold = fold['query_features']
                query_y_fold = fold['query_targets']

                print(f'\n--- Fold {fold_idx + 1}/{num_folds} ---')
                print(f'Support: {support_x.shape[0]}, Query: {query_x_fold.shape[0]}')

                with torch.no_grad():
                    supp_emb = encoder(support_x, masks=None)
                    if projection_type == 'transformer':
                        supp_emb = proto_projection(supp_emb)
                    else:
                        supp_emb = proto_projection(supp_emb.mean(dim=1))

                y_true_f, y_pred_f = [], []
                mse_total = mae_total = num_instances = 0.0
                for start in range(0, query_x_fold.shape[0], args.batch_size):
                    xb = query_x_fold[start:start + args.batch_size].to(device)
                    yb = query_y_fold[start:start + args.batch_size].to(device)
                    with torch.no_grad():
                        q_emb = encoder(xb, masks=None)
                        if projection_type == 'transformer':
                            q_emb = proto_projection(q_emb)
                        else:
                            q_emb = proto_projection(q_emb.mean(dim=1))
                        dists = -torch.cdist(q_emb, supp_emb) / dist_temperature
                        weights = torch.softmax(dists, dim=1)
                        outputs = weights @ support_y
                    mse_total += ((outputs - yb) ** 2).sum().item()
                    mae_total += torch.abs(outputs - yb).sum().item()
                    num_instances += yb.size(0)
                    y_true_f.extend(yb.cpu().numpy().flatten().tolist())
                    y_pred_f.extend(outputs.cpu().numpy().flatten().tolist())

                mse = mse_total / num_instances
                mae = mae_total / num_instances
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true_f, y_pred_f)

                fold_mses.append(mse)
                fold_maes.append(mae)
                fold_rmses.append(rmse)
                fold_r2s.append(r2)
                all_y_true.extend(y_true_f)
                all_y_pred.extend(y_pred_f)

                # Guardar por fold en results_per_task (repeat = fold_idx+1)
                results_per_task.append({
                    'task': task_name, 'repeat': fold_idx + 1,
                    'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2,
                    'support_size': support_x.shape[0],
                    'query_size': int(num_instances),
                })

            if fold_r2s:
                print(f'\n  Task {task_name} CV summary '
                      f'({len(fold_r2s)} folds):')
                print(f'  R2:  {np.mean(fold_r2s):.6f} ± {np.std(fold_r2s):.6f}')
                print(f'  RMSE: {np.mean(fold_rmses):.6f} ± {np.std(fold_rmses):.6f}')
                predictions_all_tasks.append(pd.DataFrame({
                    'task': task_name, 'repeat': 'cv',
                    'y_true': all_y_true, 'y_pred': all_y_pred,
                }))

        else:
            for repeat in range(1, n_repeats + 1):
                print(f'\n--- Repeat {repeat}/{n_repeats} ---')

                # 1) Fixed support set
                data = task.sample_fixed(k_spt, k_qry)
                support_x = data['support_features']
                support_y = data['support_targets']
                print(f'Support: {support_x.shape[0]}, Full query: {task.query_x.shape[0]}')

                # 2) Encode support → prototypes (once)
                with torch.no_grad():
                    supp_emb = encoder(support_x, masks=None)
                    if projection_type == 'transformer':
                        supp_emb = proto_projection(supp_emb)
                    else:
                        supp_emb = proto_projection(supp_emb.mean(dim=1))

                # 3) Predict over FULL query set
                y_true, y_pred = [], []
                mse_total, mae_total, num_instances = 0.0, 0.0, 0

                query_dl = task.query_dataloader(batch_size=args.batch_size)
                with torch.no_grad():
                    for x, y, idx in query_dl:
                        x, y = x.to(device), y.to(device)
                        q_emb = encoder(x, masks=None)
                        if projection_type == 'transformer':
                            q_emb = proto_projection(q_emb)
                        else:
                            q_emb = proto_projection(q_emb.mean(dim=1))

                        dists = -torch.cdist(q_emb, supp_emb) / dist_temperature
                        weights = torch.softmax(dists, dim=1)
                        outputs = weights @ support_y

                        mse_total += ((outputs - y) ** 2).sum().item()
                        mae_total += torch.abs(outputs - y).sum().item()
                        num_instances += y.size(0)
                        y_true.extend(y.cpu().numpy().flatten().tolist())
                        y_pred.extend(outputs.cpu().numpy().flatten().tolist())

                mse = mse_total / num_instances
                mae = mae_total / num_instances
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)

                print(f'\n  Repeat {repeat} Results for {task_name}:')
                print(f'  MSE: {mse:.6f} | MAE: {mae:.6f} | '
                      f'RMSE: {rmse:.6f} | R2: {r2:.6f}')

                results_per_task.append({
                    'task': task_name, 'repeat': repeat,
                    'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2,
                    'support_size': support_x.shape[0],
                    'query_size': num_instances,
                })
                predictions_all_tasks.append(pd.DataFrame({
                    'task': task_name, 'repeat': repeat,
                    'y_true': y_true, 'y_pred': y_pred,
                }))

    # -----------------------------------------------------------------------
    # Aggregation — identical to main_finetune_gadf2d.py:854-927
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(results_per_task)
    if results_df.empty:
        print('WARNING: No task results — all tasks were skipped.')
        return

    metric_cols = ['mse', 'mae', 'rmse', 'r2']
    summary_rows = []

    for tname in results_df['task'].unique():
        task_data = results_df[results_df['task'] == tname]
        mean_row = {'task': tname, 'repeat': 'MEAN'}
        std_row = {'task': tname, 'repeat': 'STD'}
        for col in metric_cols:
            mean_row[col] = task_data[col].mean()
            std_row[col] = task_data[col].std()
        mean_row['support_size'] = task_data['support_size'].iloc[0]
        mean_row['query_size'] = task_data['query_size'].iloc[0]
        std_row['support_size'] = task_data['support_size'].iloc[0]
        std_row['query_size'] = task_data['query_size'].iloc[0]
        summary_rows.extend([mean_row, std_row])

    mean_rows = [r for r in summary_rows if r['repeat'] == 'MEAN']
    mean_df = pd.DataFrame(mean_rows)
    avg_mean = {'task': 'AVERAGE', 'repeat': 'MEAN'}
    avg_std = {'task': 'AVERAGE', 'repeat': 'STD'}
    for col in metric_cols:
        avg_mean[col] = mean_df[col].mean()
        avg_std[col] = mean_df[col].std()
    avg_mean['support_size'] = mean_df['support_size'].mean()
    avg_mean['query_size'] = mean_df['query_size'].mean()
    avg_std['support_size'] = mean_df['support_size'].mean()
    avg_std['query_size'] = mean_df['query_size'].mean()
    summary_rows.extend([avg_mean, avg_std])

    final_df = pd.concat([results_df, pd.DataFrame(summary_rows)], ignore_index=True)
    final_df.to_csv(os.path.join(args.save_path, 'results_per_task.csv'), index=False)

    # per-repeat/fold summary
    if eval_mode == 'cv':
        fold_ids = sorted(results_df['repeat'].unique())
        repeat_summary_rows = []
        for rep in fold_ids:
            rep_data = results_df[results_df['repeat'] == rep]
            row = {'fold': rep, 'n_tasks': len(rep_data)}
            for col in metric_cols:
                row[col] = rep_data[col].mean()
            repeat_summary_rows.append(row)
        repeat_summary_df = pd.DataFrame(repeat_summary_rows)
        mean_row_rep = {'fold': 'MEAN', 'n_tasks': repeat_summary_df['n_tasks'].mean()}
        for col in metric_cols:
            mean_row_rep[col] = repeat_summary_df[col].mean()
        repeat_summary_df = pd.concat([repeat_summary_df, pd.DataFrame([mean_row_rep])],
                                      ignore_index=True)
        repeat_summary_df.to_csv(os.path.join(args.save_path, 'results_per_fold.csv'),
                                 index=False)
    else:
        repeat_summary_rows = []
        for rep in range(1, n_repeats + 1):
            rep_data = results_df[results_df['repeat'] == rep]
            row = {'repeat': rep}
            for col in metric_cols:
                row[col] = rep_data[col].mean()
            repeat_summary_rows.append(row)
        repeat_summary_df = pd.DataFrame(repeat_summary_rows)
        mean_row_rep = {'repeat': 'MEAN'}
        for col in metric_cols:
            mean_row_rep[col] = repeat_summary_df[col].mean()
        repeat_summary_df = pd.concat([repeat_summary_df, pd.DataFrame([mean_row_rep])],
                                      ignore_index=True)
        repeat_summary_df.to_csv(os.path.join(args.save_path, 'results_per_repeat.csv'),
                                 index=False)

    if predictions_all_tasks:
        predictions_df = pd.concat(predictions_all_tasks, ignore_index=True)
        predictions_df.to_csv(os.path.join(args.save_path, 'predictions_all_tasks.csv'),
                              index=False)

    print(f'\n{"="*60}')
    print('OVERALL RESULTS:')
    print(f'{"="*60}')
    print(f'Average MSE:  {avg_mean["mse"]:.6f} +/- {avg_std["mse"]:.6f}')
    print(f'Average MAE:  {avg_mean["mae"]:.6f} +/- {avg_std["mae"]:.6f}')
    print(f'Average RMSE: {avg_mean["rmse"]:.6f} +/- {avg_std["rmse"]:.6f}')
    print(f'Average R2:   {avg_mean["r2"]:.6f} +/- {avg_std["r2"]:.6f}')
    print(f'\nResults saved to {args.save_path}')

    if use_wandb:
        wandb.log({
            'eval/r2_mean': avg_mean['r2'],
            'eval/r2_std': avg_std['r2'],
            'eval/rmse_mean': avg_mean['rmse'],
            'eval/rmse_std': avg_std['rmse'],
            'eval/mae_mean': avg_mean['mae'],
            'eval/mae_std': avg_std['mae'],
            'eval/mse_mean': avg_mean['mse'],
            'eval/n_tasks': len(mean_rows),
            'eval/k_spt': k_spt,
            'eval/split': args.split,
        })
        try:
            art = wandb.Artifact('eval-results', type='results')
            art.add_file(os.path.join(args.save_path, 'results_per_task.csv'))
            art.add_file(os.path.join(args.save_path, 'predictions_all_tasks.csv'))
            wandb.log_artifact(art)
        except Exception as e:
            print(f'[wandb] Could not log artifact: {e}')
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ProtoNet zero-shot evaluation with joint pretrained checkpoint')

    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to joint pretraining .pth.tar checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config used during joint pretraining')
    parser.add_argument('--save_path', type=str, required=True,
                        help='Directory to save results')

    # Data / split
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate (default: test)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to downstream tasks (default: from config protonet.data_path)')

    # Few-shot settings
    parser.add_argument('--k_spt', type=int, default=None,
                        help='Support set size (default: from config protonet.k_spt)')
    parser.add_argument('--k_qry', type=int, default=None,
                        help='Query set size for fixed_val lookup (default: from config)')
    parser.add_argument('--dist_temperature', type=float, default=None,
                        help='ProtoNet distance temperature (default: from config)')

    # Inference
    parser.add_argument('--n_repeats', type=int, default=1,
                        help='Number of evaluation repeats (default: 1)')
    parser.add_argument('--eval_mode', type=str, default='fixed',
                        choices=['fixed', 'cv'],
                        help='fixed: usa fixed_val_*shots.csv (legacy); '
                             'cv: K-fold CV sobre pool unificado support∪query')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for query_dataloader (default: 64)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device (default: cuda:0)')

    # GADF normalization
    parser.add_argument('--gadf_norm_stats', type=str, default=None,
                        help='Path to GADF norm stats JSON (default: from config)')
    parser.add_argument('--gadf_global_stats', type=str, default=None,
                        help='Path to global PAA stats JSON (diagonal mode)')
    parser.add_argument('--encode_diagonal', action='store_true',
                        help='Use diagonal encoding (requires --gadf_global_stats)')

    # W&B
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='W&B project name (optional)')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='W&B run name (optional)')

    args = parser.parse_args()
    main(args)
