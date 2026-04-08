# ijepa_spectra2d — ProtoNet few-shot regression on I-JEPA embeddings
#
# Uses precomputed I-JEPA embeddings (emb_supp.pt / emb_query.pt) as input
# to a regression ProtoNet with a learnable projection network.
#
# The I-JEPA encoder is frozen — only the lightweight ProjectionNet is trained.
#
# Usage:
#   # 1. Precompute embeddings (once):
#   python scripts/precompute_embeddings.py \
#       --checkpoint logs/gadf_soil_nir/gadf_jepa-latest.pth.tar
#
#   # 2. Train ProtoNet:
#   python main_protonet_gadf2d.py \
#       --data_path data/Soil_NIR_AGG_mixed_gadf2d \
#       --episodes 5000 --repeats 3

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

from src.datasets.embedding_dataset import EmbeddingDataset
from src.protonet import EmbeddingProtoNet, EmbResNet1D, IdentityNet, ProjectionNet

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# ============================================================================
# Evaluation helper
# ============================================================================

def evaluate_split(protonet, dataset, k_spt, k_qry, fixed=True):
    """Evaluate ProtoNet on all tasks in a split.

    Returns per-task and aggregated metrics.
    """
    results = []

    for task in dataset:
        # Sample support set
        if fixed:
            sampled = task.sample_fixed(k_spt, k_qry)
        else:
            sampled = task.sample(k_spt, k_qry)

        x_spt = sampled['support_features']
        y_spt = sampled['support_targets']

        # Evaluate on full query set
        y_true_all = []
        y_pred_all = []
        y_idx_all = []

        query_dl = task.query_dataloader(batch_size=64)
        for x_q, y_q, idx in query_dl:
            x_q = x_q.to(protonet.dev)
            y_q = y_q.to(protonet.dev)
            _, _, _, _, preds = protonet.evaluate(x_spt, y_spt, x_q, y_q)
            y_true_all.extend(y_q.cpu().numpy().ravel().tolist())
            y_pred_all.extend(preds.ravel().tolist())
            y_idx_all.extend(idx.numpy().tolist())

        y_true_arr = np.array(y_true_all)
        y_pred_arr = np.array(y_pred_all)

        mse = np.mean((y_true_arr - y_pred_arr) ** 2)
        mae = np.mean(np.abs(y_true_arr - y_pred_arr))
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true_arr, y_pred_arr) if len(y_true_arr) > 1 else 0.0

        results.append({
            'task': task.name,
            'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2,
            'y_true': y_true_all, 'y_pred': y_pred_all, 'y_idx': y_idx_all,
        })

    # Aggregate
    agg = {
        'mse': np.mean([r['mse'] for r in results]),
        'mae': np.mean([r['mae'] for r in results]),
        'rmse': np.mean([r['rmse'] for r in results]),
        'r2': np.mean([r['r2'] for r in results]),
        'n_tasks': len(results),
        'per_task': results,
    }
    return agg


# ============================================================================
# Main
# ============================================================================

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    # Save config
    with open(os.path.join(args.output, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # W&B init
    use_wandb = HAS_WANDB and args.wandb_project
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args),
            dir=args.output,
        )

    all_results = []

    for rep in range(args.repeats):
        print(f'\n{"="*60}')
        print(f'Repeat {rep + 1}/{args.repeats}')
        print(f'{"="*60}')

        # Create model
        if args.arch == 'resnet1d':
            in_channels = 196 if args.emb_mode == 'patches' else 1
            projection = EmbResNet1D(
                in_channels=in_channels,
                output_dim=args.resnet_dim,
            )
        elif args.no_projection:
            projection = IdentityNet()
        else:
            projection = ProjectionNet(
                input_dim=args.embed_dim,
                hidden_dim=args.hidden_dim,
                output_dim=args.proj_dim,
            )
        protonet = EmbeddingProtoNet(
            projection=projection,
            device=device,
            dist_temperature=args.dist_temp,
            lr=args.lr,
        )

        n_params = sum(p.numel() for p in projection.parameters())
        print(f'{type(projection).__name__}: {n_params:,} params')

        # Load data
        train_data = EmbeddingDataset(
            args.data_path, split='train', device=device, scale_y=True,
            emb_mode=args.emb_mode)
        val_data = EmbeddingDataset(
            args.data_path, split='val', device=device, scale_y=True,
            emb_mode=args.emb_mode)
        test_data = EmbeddingDataset(
            args.data_path, split='test', device=device, scale_y=True,
            emb_mode=args.emb_mode)

        # Training loop
        history = {'train': {'mse': {}, 'r2': {}},
                   'val': {'mse': {}, 'r2': {}}}
        best_val_r2 = -float('inf')
        best_model_path = os.path.join(args.output, f'best_model_{rep}.pth')

        # Track last val metrics for logging between val evaluations
        last_val_mse = 0.0
        last_val_r2 = 0.0

        for episode in range(args.episodes):
            # --- Train: iterate all tasks ---
            mses, maes, rmses, r2s = [], [], [], []
            for i in range(len(train_data)):
                task = train_data[i]
                sampled = task.sample(args.k_spt, args.k_qry)
                mae, mse, rmse, r2, _ = protonet.train_step(
                    sampled['support_features'],
                    sampled['support_targets'],
                    sampled['query_features'],
                    sampled['query_targets'],
                )
                mses.append(mse)
                maes.append(mae)
                rmses.append(rmse)
                r2s.append(r2)

            train_mse = np.mean(mses)
            train_r2 = np.mean(r2s)
            history['train']['mse'][episode] = train_mse
            history['train']['r2'][episode] = train_r2

            print(f'Episode {episode:4d} | '
                  f'MSE: {train_mse:.4f} | MAE: {np.mean(maes):.4f} | '
                  f'RMSE: {np.mean(rmses):.4f} | R2: {train_r2:.4f}',
                  flush=True)

            # --- Validate ---
            if episode % args.val_freq == 0 or episode == args.episodes - 1:
                val_metrics = evaluate_split(
                    protonet, val_data, args.k_spt, args.k_qry, fixed=True)
                last_val_mse = val_metrics['mse']
                last_val_r2 = val_metrics['r2']

                history['val']['mse'][episode] = last_val_mse
                history['val']['r2'][episode] = last_val_r2

                print(f'  Val  | MSE: {last_val_mse:.4f} | '
                      f'RMSE: {val_metrics["rmse"]:.4f} | '
                      f'R2: {last_val_r2:.4f} | '
                      f'tasks: {val_metrics["n_tasks"]}', flush=True)

                if last_val_r2 > best_val_r2:
                    best_val_r2 = last_val_r2
                    protonet.save(best_model_path)
                    print(f'  -> New best val R2: {best_val_r2:.4f}')

            # W&B logging
            if use_wandb:
                log_dict = {
                    f'train/mse_rep{rep}': train_mse,
                    f'train/mae_rep{rep}': np.mean(maes),
                    f'train/rmse_rep{rep}': np.mean(rmses),
                    f'train/r2_rep{rep}': train_r2,
                    f'val/mse_rep{rep}': last_val_mse,
                    f'val/r2_rep{rep}': last_val_r2,
                    'episode': episode,
                }
                wandb.log(log_dict, step=episode + rep * args.episodes)

        # --- Test with best model ---
        protonet.load(best_model_path)
        test_metrics = evaluate_split(
            protonet, test_data, args.k_spt, args.k_qry, fixed=True)

        print(f'\nTest | MSE: {test_metrics["mse"]:.4f} | '
              f'RMSE: {test_metrics["rmse"]:.4f} | '
              f'R2: {test_metrics["r2"]:.4f} | '
              f'tasks: {test_metrics["n_tasks"]}')

        # Save predictions
        preds_rows = []
        for r in test_metrics['per_task']:
            for yt, yp, yi in zip(r['y_true'], r['y_pred'], r['y_idx']):
                preds_rows.append({
                    'y_true': yt, 'y_pred': yp, 'idx': yi, 'name': r['task']
                })
        preds_df = pd.DataFrame(preds_rows)
        preds_df.to_csv(
            os.path.join(args.output, f'predictions_test_{rep}.csv'),
            index=False)

        # Per-task test results
        per_task_df = pd.DataFrame([
            {'task': r['task'], 'mse': r['mse'], 'mae': r['mae'],
             'rmse': r['rmse'], 'r2': r['r2']}
            for r in test_metrics['per_task']
        ])
        per_task_df.to_csv(
            os.path.join(args.output, f'per_task_test_{rep}.csv'),
            index=False)

        all_results.append({
            'mse_val': best_val_r2,
            'mse_test': test_metrics['mse'],
            'mae_test': test_metrics['mae'],
            'rmse_test': test_metrics['rmse'],
            'r2_val': best_val_r2,
            'r2_test': test_metrics['r2'],
        })

        # Save history
        with open(os.path.join(args.output, f'history_{rep}.json'), 'w') as f:
            json.dump({k: {str(kk): float(vv) for kk, vv in v.items()}
                       for k, outer in history.items()
                       for k, v in outer.items()}, f)

        # Plot history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        eps_train = list(history['train']['mse'].keys())
        eps_val = list(history['val']['mse'].keys())

        axes[0].plot(eps_train, [history['train']['mse'][e] for e in eps_train],
                     label='Train')
        axes[0].plot(eps_val, [history['val']['mse'][e] for e in eps_val],
                     'o-', label='Val')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('MSE')
        axes[0].legend()
        axes[0].set_title('MSE')

        axes[1].plot(eps_train, [history['train']['r2'][e] for e in eps_train],
                     label='Train')
        axes[1].plot(eps_val, [history['val']['r2'][e] for e in eps_val],
                     'o-', label='Val')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('R2')
        axes[1].legend()
        axes[1].set_title('R2')

        plt.tight_layout()
        plt.savefig(os.path.join(args.output, f'history_{rep}.png'), dpi=100)
        plt.close()

    # Aggregate results across repeats
    results_df = pd.DataFrame(all_results)
    results_df.loc['mean'] = results_df.mean()
    results_df.to_csv(os.path.join(args.output, 'results.csv'))
    print(f'\n{"="*60}')
    print('Final results (mean over repeats):')
    print(results_df.loc['mean'].to_string())
    print(f'{"="*60}')

    if use_wandb:
        wandb.summary['test/r2_mean'] = results_df.loc['mean']['r2_test']
        wandb.summary['test/rmse_mean'] = results_df.loc['mean']['rmse_test']
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='ProtoNet few-shot regression on I-JEPA embeddings')

    # Data
    parser.add_argument('--data_path', type=str,
                        default='data/Soil_NIR_AGG_mixed_gadf2d')
    parser.add_argument('--output', type=str,
                        default='results/protonet_gadf2d/')

    # Training
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--k_spt', type=int, default=25)
    parser.add_argument('--k_qry', type=int, default=25)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--val_freq', type=int, default=50)

    # Model
    parser.add_argument('--embed_dim', type=int, default=768,
                        help='I-JEPA embedding dimension')
    parser.add_argument('--proj_dim', type=int, default=128,
                        help='Projection output dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Projection hidden dimension')
    parser.add_argument('--dist_temp', type=float, default=0.5,
                        help='Distance temperature for softmax')
    parser.add_argument('--no_projection', action='store_true',
                        help='Skip projection (use raw embeddings)')
    parser.add_argument('--arch', type=str, default='mlp',
                        choices=['mlp', 'resnet1d'],
                        help='Projection architecture: mlp (default) or resnet1d '
                             '(convolutional, ported from fewshot_nir_orig)')
    parser.add_argument('--emb_mode', type=str, default='mean',
                        choices=['mean', 'patches'],
                        help='mean: load emb_supp.pt [N,D]; '
                             'patches: load emb_supp_patches.pt [N,P,D]')
    parser.add_argument('--resnet_dim', type=int, default=512,
                        help='EmbResNet1D output dimension (default 512)')

    # W&B
    parser.add_argument('--wandb_project', type=str, default='protonet-gadf2d')
    parser.add_argument('--wandb_name', type=str, default=None)

    # Device
    parser.add_argument('--device', type=str, default='cuda')

    main(parser.parse_args())
