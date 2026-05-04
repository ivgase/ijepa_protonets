# ProtoNet-only fine-tuning starting from a joint checkpoint.
#
# Loads the target_encoder from a joint (I-JEPA + ProtoNet) checkpoint and
# fine-tunes it for a few epochs with plain episodic ProtoNet training:
#   - one encoder, no EMA, no predictor, no mask collator, no I-JEPA loss
#   - fresh AdamW optimizer with cosine LR schedule
#   - lambda_proto = 1.0 (single objective)
#
# Checkpoints are saved with both 'target_encoder' and 'encoder' keys so
# that main_finetune_gadf2d.py and main_eval_joint_protonet.py can consume
# them without modification.

import os

try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
import math
import sys
import time
import yaml
from datetime import datetime, timedelta

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as vit
from src.train_joint import EpisodicTaskSampler, ProtoNetEvaluator
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import CSVLogger, gpu_timer, AverageMeter
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.utils.tensors import trunc_normal_

# --
log_freq = 10
checkpoint_freq = 10   # smaller than joint (suitable for short runs)
artifact_ttl = timedelta(days=15)
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Final evaluation on a split (test / val / train)
# ---------------------------------------------------------------------------

def evaluate_on_split(
    encoder,
    proto_projection,
    *,
    data_path,
    split,
    save_path,
    device,
    crop_size,
    norm_stats,
    global_stats,
    k_spt,
    k_qry,
    dist_temperature,
    n_repeats,
    batch_size,
    projection_type,
    use_wandb,
):
    """Run ProtoNet inference over every task in ``split`` and save results.

    Mirrors ``main_eval_joint_protonet.py``: fixed_val support set,
    full query via ``query_dataloader``, same metrics (mse/mae/rmse/r2),
    same aggregation, same output files.
    """
    import pandas as pd
    from sklearn.metrics import r2_score

    # Local import to avoid circular dependency with main_finetune_gadf2d
    from main_finetune_gadf2d import SimpleTask2D

    os.makedirs(save_path, exist_ok=True)

    # Unwrap DDP
    enc = encoder.module if hasattr(encoder, 'module') else encoder
    enc.eval()
    proj = None
    if proto_projection is not None:
        proj = proto_projection.module if hasattr(proto_projection, 'module') else proto_projection
        proj.eval()

    # -- Load tasks from split
    splits_file = os.path.join(data_path, 'splits.csv')
    if not os.path.exists(splits_file):
        logger.warning(f'splits.csv not found at {data_path} — skipping final eval')
        return None

    splits_df = pd.read_csv(splits_file, index_col=0)
    task_names = sorted(splits_df[splits_df['split'] == split]['task'].tolist())
    logger.info(f'[eval] Loading {len(task_names)} tasks from split "{split}"...')

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
            logger.warning(f'[eval]  Skipping {tname}: {e}')

    logger.info(f'[eval] Loaded {len(tasks)} tasks')
    if not tasks:
        logger.warning('[eval] No tasks available — skipping')
        return None

    # Check fixed_val availability for requested k_spt
    _supp_csv = os.path.join(tasks[0].data_path,
                             f'fixed_val_support_{k_spt}shots.csv')
    if not os.path.exists(_supp_csv):
        available = sorted({
            int(f.split('_')[3].replace('shots.csv', ''))
            for f in os.listdir(tasks[0].data_path)
            if f.startswith('fixed_val_support_') and f.endswith('shots.csv')
        })
        logger.error(
            f'[eval] fixed_val_support_{k_spt}shots.csv not found. '
            f'Available shot counts: {available}. Skipping final eval.'
        )
        return None

    # -- Task loop
    results_per_task = []
    predictions_all_tasks = []

    logger.info(f'[eval] Running {n_repeats} repeat(s) per task')
    for task_idx, task in enumerate(tasks):
        task_name = task.name
        logger.info(f'[eval] {"="*60}')
        logger.info(f'[eval] Task {task_idx + 1}/{len(tasks)}: {task_name}')
        logger.info(f'[eval] {"="*60}')

        for repeat in range(1, n_repeats + 1):
            # 1) Fixed support set
            data = task.sample_fixed(k_spt, k_qry)
            support_x = data['support_features']  # [k_spt, 1, H, W]
            support_y = data['support_targets']   # [k_spt, 1]
            logger.info(
                f'[eval]   Repeat {repeat}/{n_repeats} | '
                f'support={support_x.shape[0]} full_query={task.query_x.shape[0]}'
            )

            # 2) Encode support once → prototypes
            with torch.no_grad():
                supp_emb = enc(support_x, masks=None)  # [k_spt, N, D]
                if proj is not None and projection_type == 'transformer':
                    supp_emb = proj(supp_emb)
                else:
                    supp_emb = supp_emb.mean(dim=1)
                    if proj is not None:
                        supp_emb = proj(supp_emb)

            # 3) Predict over FULL query set
            y_true, y_pred = [], []
            mse_total, mae_total, num_instances = 0.0, 0.0, 0

            query_dl = task.query_dataloader(batch_size=batch_size)
            with torch.no_grad():
                for x, y, _idx in query_dl:
                    x, y = x.to(device), y.to(device)
                    q_emb = enc(x, masks=None)
                    if proj is not None and projection_type == 'transformer':
                        q_emb = proj(q_emb)
                    else:
                        q_emb = q_emb.mean(dim=1)
                        if proj is not None:
                            q_emb = proj(q_emb)

                    dists = -torch.cdist(q_emb, supp_emb) / dist_temperature
                    weights = torch.softmax(dists, dim=1)
                    outputs = weights @ support_y  # [B, 1]

                    mse_total += ((outputs - y) ** 2).sum().item()
                    mae_total += torch.abs(outputs - y).sum().item()
                    num_instances += y.size(0)
                    y_true.extend(y.cpu().numpy().flatten().tolist())
                    y_pred.extend(outputs.cpu().numpy().flatten().tolist())

            mse = mse_total / num_instances
            mae = mae_total / num_instances
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred)

            logger.info(
                f'[eval]   {task_name} rep {repeat}: '
                f'MSE={mse:.6f}  MAE={mae:.6f}  RMSE={rmse:.6f}  R2={r2:.6f}'
            )

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

    # -- Aggregation (identical to main_eval_joint_protonet.py)
    results_df = pd.DataFrame(results_per_task)
    if results_df.empty:
        logger.warning('[eval] No task results — all tasks were skipped.')
        return None

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
    final_df.to_csv(os.path.join(save_path, 'results_per_task.csv'), index=False)

    # Per-repeat summary
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
    repeat_summary_df = pd.concat(
        [repeat_summary_df, pd.DataFrame([mean_row_rep])],
        ignore_index=True,
    )
    repeat_summary_df.to_csv(os.path.join(save_path, 'results_per_repeat.csv'),
                             index=False)

    if predictions_all_tasks:
        predictions_df = pd.concat(predictions_all_tasks, ignore_index=True)
        predictions_df.to_csv(os.path.join(save_path, 'predictions_all_tasks.csv'),
                              index=False)

    logger.info(f'[eval] {"="*60}')
    logger.info('[eval] OVERALL RESULTS:')
    logger.info(f'[eval] {"="*60}')
    logger.info(f'[eval] Average MSE:  {avg_mean["mse"]:.6f} +/- {avg_std["mse"]:.6f}')
    logger.info(f'[eval] Average MAE:  {avg_mean["mae"]:.6f} +/- {avg_std["mae"]:.6f}')
    logger.info(f'[eval] Average RMSE: {avg_mean["rmse"]:.6f} +/- {avg_std["rmse"]:.6f}')
    logger.info(f'[eval] Average R2:   {avg_mean["r2"]:.6f} +/- {avg_std["r2"]:.6f}')
    logger.info(f'[eval] Results saved to {save_path}')

    summary = {
        'eval/r2_mean': avg_mean['r2'],
        'eval/r2_std': avg_std['r2'],
        'eval/rmse_mean': avg_mean['rmse'],
        'eval/rmse_std': avg_std['rmse'],
        'eval/mae_mean': avg_mean['mae'],
        'eval/mae_std': avg_std['mae'],
        'eval/mse_mean': avg_mean['mse'],
        'eval/n_tasks': len(mean_rows),
        'eval/k_spt': k_spt,
        'eval/split': split,
    }

    if use_wandb:
        try:
            import wandb
            wandb.log(summary)
            art = wandb.Artifact('eval-results', type='results')
            art.add_file(os.path.join(save_path, 'results_per_task.csv'))
            art.add_file(os.path.join(save_path, 'results_per_repeat.csv'))
            pred_csv = os.path.join(save_path, 'predictions_all_tasks.csv')
            if os.path.exists(pred_csv):
                art.add_file(pred_csv)
            wandb.log_artifact(art)
        except Exception as e:
            logger.warning(f'[eval] W&B logging failed: {e}')

    return summary


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta'].get('load_checkpoint', False) or resume_preempt
    r_file = args['meta'].get('read_checkpoint', None)

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    crop_size = args['data']['crop_size']

    # -- MASK (only need patch_size for model construction)
    patch_size = args['mask']['patch_size']

    # -- OPTIMIZATION
    ipe_scale = args['optimization'].get('ipe_scale', 1.0)
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    base_folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    folder = os.path.join(base_folder, f'{tag}_{timestamp}')
    os.makedirs(folder, exist_ok=True)

    # -- WANDB
    wandb_cfg = args.get('wandb', {})
    use_wandb = wandb_cfg.get('enable', False)
    wandb_project = wandb_cfg.get('project', 'ijepa-gadf2d-joint')
    wandb_name = wandb_cfg.get('name', None)

    # -- PROBE (ProtoNet online evaluation during training)
    probe_cfg = args.get('probe', {})
    probe_freq = probe_cfg.get('freq', 0)
    probe_data_path = probe_cfg.get('data_path', '')
    probe_k_spt = probe_cfg.get('k_spt', 25)
    probe_k_qry = probe_cfg.get('k_qry', 25)

    # -- EVAL (final evaluation on a task split after training)
    eval_cfg = args.get('eval', {})
    eval_enable = eval_cfg.get('enable', True)
    eval_split = eval_cfg.get('split', 'test')
    eval_n_repeats = int(eval_cfg.get('n_repeats', 1))
    eval_batch_size = int(eval_cfg.get('batch_size', 64))
    eval_save_subdir = eval_cfg.get('save_subdir', None)  # None → <folder>/eval_<split>/

    # -- PROTONET (training)
    proto_cfg = args.get('protonet', {})
    proto_data_path = proto_cfg.get('data_path', '')
    proto_k_spt = proto_cfg.get('k_spt', 25)
    proto_k_qry = proto_cfg.get('k_qry', 25)
    proto_dist_temp = float(proto_cfg.get('dist_temperature', 0.5))
    proto_loss_type = str(proto_cfg.get('loss_type', 'mse')).lower()
    proto_use_projection = proto_cfg.get('use_projection', False)
    proto_projection_type = proto_cfg.get('projection_type', 'mlp')
    proto_hidden_dim = proto_cfg.get('projection_hidden_dim', 256)
    proto_output_dim = proto_cfg.get('projection_output_dim', 128)
    aggregator_num_layers = int(proto_cfg.get('aggregator_num_layers', 1))
    aggregator_num_heads = int(proto_cfg.get('aggregator_num_heads', 4))
    aggregator_mlp_ratio = float(proto_cfg.get('aggregator_mlp_ratio', 2.0))
    aggregator_dropout = float(proto_cfg.get('aggregator_dropout', 0.1))
    aggregator_attn_dropout = float(proto_cfg.get('aggregator_attn_dropout', 0.1))
    meta_batch_size = int(proto_cfg.get('meta_batch_size', 8))

    valid_proto_losses = {'mse', 'smooth_l1'}
    if proto_loss_type not in valid_proto_losses:
        raise ValueError(
            f'Invalid protonet.loss_type={proto_loss_type!r}. '
            f'Expected one of {sorted(valid_proto_losses)}'
        )

    # Save params for reproducibility
    dump = os.path.join(folder, 'params-proto-only.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)

    # ----------------------------------------------------------------------- #

    try:
        import torch.multiprocessing as mp
        mp.set_start_method('spawn')
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- W&B (rank 0 only)
    if use_wandb and rank == 0:
        try:
            import wandb
            if wandb_name is None:
                wandb_name = f'{tag}_{model_name}_proto_only'
            wandb.init(project=wandb_project, name=wandb_name, config=args)
            wandb.config.update({'checkpoint_dir': os.path.abspath(folder)})
        except Exception as e:
            logger.warning(f'W&B init failed ({e}). Disabling wandb.')
            use_wandb = False
    else:
        use_wandb = use_wandb and (rank == 0)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    save_path = os.path.join(folder, f'{tag}' + '-ep{epoch}.pth.tar')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    load_path = None
    if load_model:
        if r_file is not None:
            load_path = r_file if os.path.isabs(r_file) else os.path.join(base_folder, r_file)

    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss_proto'),
        ('%d', 'time (ms)'),
    )

    # ----------------------------------------------------------------------- #
    # Model
    # ----------------------------------------------------------------------- #

    def init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        in_chans=1,
    )
    for m in encoder.modules():
        init_weights(m)
    encoder.to(device)
    logger.info(f'Encoder: {sum(p.numel() for p in encoder.parameters()):,} params')

    # -- Proto projection (optional; None for current checkpoint)
    proto_projection = None
    if proto_use_projection:
        if proto_projection_type == 'transformer':
            from src.protonet import TransformerAggregator
            num_patches = (crop_size // patch_size) ** 2
            proto_projection = TransformerAggregator(
                input_dim=encoder.embed_dim,
                d_model=proto_hidden_dim,
                output_dim=proto_output_dim,
                num_patches=num_patches,
                num_layers=aggregator_num_layers,
                num_heads=aggregator_num_heads,
                mlp_ratio=aggregator_mlp_ratio,
                dropout=aggregator_dropout,
                attn_dropout=aggregator_attn_dropout,
            ).to(device)
        else:
            from src.protonet import ProjectionNet
            proto_projection = ProjectionNet(
                input_dim=encoder.embed_dim,
                hidden_dim=proto_hidden_dim,
                output_dim=proto_output_dim,
            ).to(device)
        logger.info(
            f'ProtoNet projection ({proto_projection_type}): '
            f'{sum(p.numel() for p in proto_projection.parameters()):,} params'
        )

    # ----------------------------------------------------------------------- #
    # GADF norm stats (needed by episodic sampler)
    # ----------------------------------------------------------------------- #

    dataset_type = args['data'].get('dataset_type', 'gadf')
    gadf_norm_stats_path = args['data'].get('gadf_norm_stats', None)
    use_diagonal = args['data'].get('gadf_global_stats', None) is not None
    if gadf_norm_stats_path is not None:
        from src.gadf_utils import load_norm_stats
        norm_stats_tuple = load_norm_stats(gadf_norm_stats_path, use_diagonal=use_diagonal)
        logger.info(
            f'GADF norm stats from {gadf_norm_stats_path} '
            f'(diagonal={use_diagonal}): '
            f'mean={norm_stats_tuple[0]}, std={norm_stats_tuple[1]}'
        )
    else:
        norm_stats_tuple = ((-0.0000,), (0.5922,))
        logger.warning('No gadf_norm_stats in config, using hardcoded defaults')

    gadf_global_stats_path = args['data'].get('gadf_global_stats', None)
    task_global_stats = None
    if gadf_global_stats_path is not None:
        from src.gadf_utils import load_global_stats
        task_global_stats = load_global_stats(gadf_global_stats_path)
        logger.info(
            f'Diagonal encoding for tasks: '
            f'min={task_global_stats[0]:.4f}, max={task_global_stats[1]:.4f}'
        )

    # ----------------------------------------------------------------------- #
    # Episodic sampler + evaluator
    # ----------------------------------------------------------------------- #

    proto_sampler = EpisodicTaskSampler(
        data_path=proto_data_path,
        image_size=crop_size,
        norm_stats=norm_stats_tuple,
        k_spt=proto_k_spt,
        k_qry=proto_k_qry,
        global_stats=task_global_stats,
    )
    if not proto_sampler.tasks:
        raise RuntimeError(
            f'No episodic tasks loaded from {proto_data_path} — nothing to train on.'
        )
    logger.info(f'ProtoNet sampler: {len(proto_sampler)} train tasks')

    proto_evaluator = None
    if probe_freq > 0 and rank == 0:
        proto_evaluator = ProtoNetEvaluator(
            data_path=probe_data_path,
            image_size=crop_size,
            norm_stats=norm_stats_tuple,
            k_spt=probe_k_spt,
            k_qry=probe_k_qry,
            dist_temperature=proto_dist_temp,
            device=device,
            global_stats=task_global_stats,
            projection_type=proto_projection_type,
        )
        if not proto_evaluator.tasks:
            proto_evaluator = None

    # ----------------------------------------------------------------------- #
    # Optimizer (fresh AdamW, no predictor, no EMA)
    # ----------------------------------------------------------------------- #

    n_proto_steps_per_epoch = math.ceil(len(proto_sampler) / meta_batch_size)
    ipe = n_proto_steps_per_epoch
    total_steps = int(ipe * num_epochs * ipe_scale)

    param_groups = [
        {
            'params': [p for n, p in encoder.named_parameters()
                       if 'bias' not in n and p.ndim != 1],
        },
        {
            'params': [p for n, p in encoder.named_parameters()
                       if 'bias' in n or p.ndim == 1],
            'WD_exclude': True,
            'weight_decay': 0.0,
        },
    ]
    if proto_projection is not None:
        no_wd_names = set()
        if proto_projection_type == 'transformer':
            no_wd_names.update({'cls_token', 'pos_embed'})
        param_groups.append({
            'params': [p for n, p in proto_projection.named_parameters()
                       if 'bias' not in n and p.ndim != 1 and n not in no_wd_names],
        })
        param_groups.append({
            'params': [p for n, p in proto_projection.named_parameters()
                       if 'bias' in n or p.ndim == 1 or n in no_wd_names],
            'WD_exclude': True,
            'weight_decay': 0.0,
        })

    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * ipe),
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=total_steps,
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=total_steps,
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None

    # ----------------------------------------------------------------------- #
    # Load checkpoint (target_encoder weights only; optimizer stays fresh)
    # ----------------------------------------------------------------------- #

    source_epoch = 0
    if load_model and load_path is not None:
        try:
            ckpt = torch.load(load_path, map_location='cpu')
            source_epoch = ckpt.get('epoch', 0)

            # Strip DDP 'module.' prefix if present
            state = {k.replace('module.', ''): v
                     for k, v in ckpt['target_encoder'].items()}
            msg = encoder.load_state_dict(state, strict=True)
            logger.info(
                f'Loaded target_encoder from epoch {source_epoch} '
                f'({load_path}): {msg}'
            )

            if proto_projection is not None and 'proto_projection' in ckpt:
                pstate = {k.replace('module.', ''): v
                          for k, v in ckpt['proto_projection'].items()}
                proto_projection.load_state_dict(pstate, strict=True)
                logger.info('Loaded proto_projection from checkpoint')
            del ckpt
        except Exception as e:
            logger.warning(f'Could not load checkpoint: {e} — starting from random weights')
    else:
        logger.info('No checkpoint loaded — training from random weights')

    # ----------------------------------------------------------------------- #
    # DDP
    # ----------------------------------------------------------------------- #

    if world_size > 1:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        if proto_projection is not None:
            proto_projection = DistributedDataParallel(proto_projection, static_graph=True)

    # ----------------------------------------------------------------------- #
    # Save checkpoint helper
    # ----------------------------------------------------------------------- #

    def save_checkpoint(epoch):
        save_dict = {
            'target_encoder': encoder.state_dict(),
            'encoder': encoder.state_dict(),  # duplicate for compat
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss_proto': loss_meter.avg,
            'lr': lr,
            'source_checkpoint': load_path,
            'source_epoch': source_epoch,
        }
        if proto_projection is not None:
            save_dict['proto_projection'] = proto_projection.state_dict()
        if rank == 0:
            torch.save(save_dict, latest_path)
            if epoch % checkpoint_freq == 0:
                ckpt_path = save_path.format(epoch=epoch)
                torch.save(save_dict, ckpt_path)
                logger.info(f'Saved checkpoint: {ckpt_path}')

    # ----------------------------------------------------------------------- #
    # Training loop
    # ----------------------------------------------------------------------- #

    best_probe_r2 = -float('inf')

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        encoder.train()
        if proto_projection is not None:
            proto_projection.train()

        loss_meter = AverageMeter()
        time_meter = AverageMeter()

        task_order = np.random.permutation(len(proto_sampler))
        task_ptr = 0
        step = 0

        # -- Proto step helper (closure over optimizer / schedulers / etc.)
        def proto_step(episodes):
            _new_lr = scheduler.step()
            _new_wd = wd_scheduler.step()
            n_tasks = len(episodes)
            loss_accum = 0.0

            for (spt_imgs, spt_y, qry_imgs, qry_y) in episodes:
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    spt_emb = encoder(spt_imgs)   # [k_spt, N_patches, D]
                    qry_emb = encoder(qry_imgs)   # [k_qry, N_patches, D]

                    if proto_projection is not None and proto_projection_type == 'transformer':
                        spt_emb = proto_projection(spt_emb)   # [k_spt, out_dim]
                        qry_emb = proto_projection(qry_emb)
                    else:
                        spt_emb = spt_emb.mean(dim=1)          # [k_spt, D]
                        qry_emb = qry_emb.mean(dim=1)
                        if proto_projection is not None:
                            spt_emb = proto_projection(spt_emb)
                            qry_emb = proto_projection(qry_emb)

                    dists = -torch.cdist(qry_emb, spt_emb) / proto_dist_temp
                    weights = torch.softmax(dists, dim=1)
                    preds = weights @ spt_y

                    if proto_loss_type == 'smooth_l1':
                        task_loss = F.smooth_l1_loss(preds, qry_y)
                    else:
                        task_loss = F.mse_loss(preds, qry_y)

                    task_loss = AllReduce.apply(task_loss)
                    task_loss_scaled = task_loss / n_tasks

                if use_bfloat16:
                    scaler.scale(task_loss_scaled).backward()
                else:
                    task_loss_scaled.backward()

                loss_accum += task_loss.item()

            if use_bfloat16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            return loss_accum / n_tasks, _new_lr, _new_wd

        while task_ptr < len(task_order):
            batch_end = min(task_ptr + meta_batch_size, len(task_order))
            batch_idx = task_order[task_ptr:batch_end]
            episodes = [
                proto_sampler.sample_episode_by_idx(int(i), device)
                for i in batch_idx
            ]

            result, etime = gpu_timer(lambda: proto_step(episodes))
            loss_val, _new_lr, _new_wd = result

            loss_meter.update(loss_val)
            time_meter.update(etime)

            if (step % log_freq == 0) or np.isnan(loss_val) or np.isinf(loss_val):
                logger.info(
                    '[%d, %5d/%d] loss_proto: %.4f '
                    '[wd: %.2e] [lr: %.2e] '
                    '[mem: %.2e] (%.1f ms)'
                    % (
                        epoch + 1, step + 1, n_proto_steps_per_epoch,
                        loss_meter.avg,
                        _new_wd, _new_lr,
                        torch.cuda.max_memory_allocated() / 1024.**2,
                        time_meter.avg,
                    )
                )

            csv_logger.log(epoch + 1, step, loss_val, etime)
            assert not np.isnan(loss_val), 'loss is nan'

            task_ptr = batch_end
            step += 1

        logger.info(
            f'Epoch {epoch + 1} done — '
            f'avg loss_proto={loss_meter.avg:.4f} '
            f'({n_proto_steps_per_epoch} steps, {time_meter.avg:.1f} ms/step)'
        )

        save_checkpoint(epoch + 1)

        # -- ProtoNet evaluation and W&B logging (rank 0)
        if rank == 0:
            wandb_log = {
                'train/loss_proto': loss_meter.avg,
                'train/lr': _new_lr,
                'train/wd': _new_wd,
                'train/time_ms': time_meter.avg,
                'epoch': epoch + 1,
            }

            if proto_evaluator is not None and (epoch + 1) % probe_freq == 0:
                encoder.eval()
                if proto_projection is not None:
                    proto_projection.eval()
                t0 = time.time()
                probe_metrics = proto_evaluator.evaluate(encoder, proto_projection, device)
                probe_time = time.time() - t0
                logger.info(
                    f'ProtoNet eval (ep {epoch + 1}): '
                    f'R²={probe_metrics["probe/r2_mean"]:.4f}±'
                    f'{probe_metrics["probe/r2_std"]:.4f}  '
                    f'RMSE={probe_metrics["probe/rmse_mean"]:.4f}±'
                    f'{probe_metrics["probe/rmse_std"]:.4f}  '
                    f'({probe_metrics["probe/n_tasks"]} tasks, {probe_time:.1f}s)'
                )
                wandb_log.update(probe_metrics)
                r2 = probe_metrics['probe/r2_mean']
                if r2 > best_probe_r2:
                    best_probe_r2 = r2
                    logger.info(f'New best probe R²={r2:.4f} at epoch {epoch + 1}')
                encoder.train()
                if proto_projection is not None:
                    proto_projection.train()

            if use_wandb:
                import wandb
                wandb.log(wandb_log)

    # -- Log final checkpoint as W&B artifact (before eval so it's saved even if eval fails)
    if use_wandb and rank == 0:
        import wandb
        art = wandb.Artifact(
            f'{tag}-checkpoint-latest',
            type='model',
            metadata={'epoch': num_epochs, 'loss': loss_meter.avg,
                      'source_epoch': source_epoch},
        )
        art.add_reference(f'file://{os.path.abspath(latest_path)}')
        try:
            wandb.log_artifact(art)
        except Exception as e:
            logger.warning(f'W&B artifact logging failed: {e}')

    # ----------------------------------------------------------------------- #
    # Final evaluation on a task split (rank 0 only)
    # ----------------------------------------------------------------------- #
    if eval_enable and rank == 0:
        eval_save = (
            os.path.join(folder, eval_save_subdir)
            if eval_save_subdir is not None
            else os.path.join(folder, f'eval_{eval_split}')
        )
        logger.info(f'Running final evaluation on split "{eval_split}" → {eval_save}')
        try:
            evaluate_on_split(
                encoder=encoder,
                proto_projection=proto_projection,
                data_path=proto_data_path,
                split=eval_split,
                save_path=eval_save,
                device=device,
                crop_size=crop_size,
                norm_stats=(norm_stats_tuple[0][0], norm_stats_tuple[1][0]),
                global_stats=task_global_stats,
                k_spt=proto_k_spt,
                k_qry=proto_k_qry,
                dist_temperature=proto_dist_temp,
                n_repeats=eval_n_repeats,
                batch_size=eval_batch_size,
                projection_type=proto_projection_type,
                use_wandb=use_wandb,
            )
        except Exception as e:
            logger.exception(f'Final evaluation failed: {e}')

    # -- Finish W&B
    if use_wandb and rank == 0:
        import wandb
        wandb.finish()

    logger.info(
        f'Training complete. '
        f'Best probe R²={best_probe_r2:.4f}. '
        f'Latest checkpoint: {latest_path}'
    )


if __name__ == '__main__':
    main()
