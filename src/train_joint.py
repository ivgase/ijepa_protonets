# Joint I-JEPA + ProtoNet pretraining
#
# Per epoch:
#   I-JEPA iterations and ProtoNet meta-batch steps are INTERLEAVED.
#   Every ~proto_freq I-JEPA iterations, one ProtoNet step processes the next
#   mini-batch of tasks. All train tasks are covered each epoch.
#
# - L_jepa: Smooth L1 (patch prediction, same as src/train.py)
# - L_proto: MSE averaged over mini-batch of episodic tasks (ProtoNet regression)
# - Lambda warmup: lambda_proto ramps from 0 to target over proto_warmup epochs
#
# The context encoder is shared by both objectives.
# The target encoder is updated via EMA (no gradients).
#
# This script is self-contained and does NOT modify src/train.py.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
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
from datetime import timedelta

import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import (
    init_distributed,
    AllReduce
)
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch
from src.datasets.gadf_dataset import make_gadf
from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
from src.sigreg import SigRegHookCollector
from src.transforms import make_transforms

# --
log_timings = True
log_freq = 10
checkpoint_freq = 50
artifact_ttl = timedelta(days=15)
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


# ---------------------------------------------------------------------------
# Episodic task sampler (for ProtoNet training branch)
# ---------------------------------------------------------------------------

class EpisodicTaskSampler:
    """Pre-loads train-split tasks and samples episodes for ProtoNet training.

    All tasks from the 'train' split of splits.csv are loaded once at init.
    Spectra are converted to GADF 2D images once and cached in CPU memory.
    Each call to ``sample_episode()`` picks a random task and draws
    k_spt / k_qry samples.
    """

    def __init__(self, data_path, image_size=224, norm_stats=None,
                 k_spt=25, k_qry=25, global_stats=None):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        self._StandardScaler = StandardScaler

        self.k_spt = k_spt
        self.k_qry = k_qry
        self.image_size = image_size
        self.norm_stats = norm_stats
        self.tasks = []  # list of (name, supp_x, supp_y, query_x, query_y)

        splits_file = os.path.join(data_path, 'splits.csv')
        if not os.path.exists(splits_file):
            logger.warning(f'No splits.csv at {data_path} — episodic sampler disabled')
            return

        splits_df = pd.read_csv(splits_file, index_col=0)
        train_tasks = sorted(splits_df[splits_df['split'] == 'train']['task'].tolist())

        logger.info(f'Loading {len(train_tasks)} episodic tasks from train split...')

        for tname in train_tasks:
            task_dir = os.path.join(data_path, tname)
            if not os.path.isdir(task_dir):
                continue
            try:
                t = self._load_task(task_dir, tname)
                if t is not None:
                    self.tasks.append(t)
            except Exception as e:
                logger.warning(f'  Skipping episodic task {tname}: {e}')

        logger.info(f'Episodic sampler ready: {len(self.tasks)} tasks loaded')

    def _load_task(self, task_dir, name):
        """Load all samples for a single task from precomputed tensors, scale y."""
        import pandas as pd

        y_supp = pd.read_csv(os.path.join(task_dir, 'y_supp.csv'), index_col=0)
        y_query = pd.read_csv(os.path.join(task_dir, 'y_query.csv'), index_col=0)

        # Use first numeric column as target
        num_cols = y_supp.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return None
        col = num_cols[0]
        ys = y_supp[col]
        yq = y_query[col]

        # Drop NaN targets
        supp_valid = ~ys.isna()
        query_valid = ~yq.isna()
        ys_np = ys[supp_valid].values.astype(np.float32)
        yq_np = yq[query_valid].values.astype(np.float32)

        if len(ys_np) < 2 or len(yq_np) < 2:
            return None

        # Scale y (fit on support)
        y_scaler = self._StandardScaler()
        ys_np = y_scaler.fit_transform(ys_np.reshape(-1, 1)).flatten()
        yq_np = y_scaler.transform(yq_np.reshape(-1, 1)).flatten()

        # Skip tasks where the query set is far outside the support distribution.
        # These tasks have extreme distribution shift (e.g. Luxembourg-CEC: 18σ) and
        # produce MSE >> 100 that destabilises training without providing useful signal.
        if np.max(np.abs(yq_np)) > 10.0:
            return None

        # Load precomputed GADF tensors [N, 1, H, W] float16 → float32
        sx = torch.load(os.path.join(task_dir, 'X_supp.pt'),
                        map_location='cpu').float()[supp_valid.values]
        qx = torch.load(os.path.join(task_dir, 'X_query.pt'),
                        map_location='cpu').float()[query_valid.values]

        # Normalize GADF
        if self.norm_stats is not None:
            mean = torch.tensor(self.norm_stats[0]).view(1, 1, 1, 1)
            std = torch.tensor(self.norm_stats[1]).view(1, 1, 1, 1)
            sx = (sx - mean) / std
            qx = (qx - mean) / std

        sy = torch.from_numpy(ys_np).unsqueeze(1)
        qy = torch.from_numpy(yq_np).unsqueeze(1)

        return (name, sx, sy, qx, qy)

    def sample_episode(self, device):
        """Sample one episode: random task, random k_spt + k_qry."""
        idx = np.random.randint(len(self.tasks))
        name, sx, sy, qx, qy = self.tasks[idx]

        n_spt = min(self.k_spt, len(sy))
        n_qry = min(self.k_qry, len(qy))
        spt_idx = np.random.choice(len(sy), n_spt, replace=False)
        qry_idx = np.random.choice(len(qy), n_qry, replace=False)

        return (
            sx[spt_idx].to(device, non_blocking=True),
            sy[spt_idx].to(device, non_blocking=True),
            qx[qry_idx].to(device, non_blocking=True),
            qy[qry_idx].to(device, non_blocking=True),
        )

    def __len__(self):
        return len(self.tasks)

    def sample_episode_by_idx(self, task_idx, device):
        """Sample one episode from a specific task index (for sequential epoch iteration)."""
        name, sx, sy, qx, qy = self.tasks[task_idx]
        n_spt = min(self.k_spt, len(sy))
        n_qry = min(self.k_qry, len(qy))
        spt_idx = np.random.choice(len(sy), n_spt, replace=False)
        qry_idx = np.random.choice(len(qy), n_qry, replace=False)
        return (
            sx[spt_idx].to(device, non_blocking=True),
            sy[spt_idx].to(device, non_blocking=True),
            qx[qry_idx].to(device, non_blocking=True),
            qy[qry_idx].to(device, non_blocking=True),
        )

    def sample_meta_batch(self, meta_batch_size, device):
        """Sample meta_batch_size episodes for the epoch-level ProtoNet step."""
        return [self.sample_episode(device) for _ in range(meta_batch_size)]


# ---------------------------------------------------------------------------
# ProtoNet evaluator (replaces LinearProbeEvaluator)
# ---------------------------------------------------------------------------

class ProtoNetEvaluator:
    """Evaluates encoder quality via ProtoNet regression on val-split tasks.

    Uses the same distance-weighted regression as fewshot_nir_orig:
      preds = softmax(-cdist(query, support) / temperature) @ y_support

    No linear head is trained — evaluation is purely based on embedding
    distances, matching the downstream use case.
    """

    def __init__(self, data_path, image_size=224, norm_stats=None,
                 k_spt=25, k_qry=25, dist_temperature=0.5,
                 device='cpu', global_stats=None, projection_type='mlp'):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        self._StandardScaler = StandardScaler

        self.k_spt = k_spt
        self.k_qry = k_qry
        self.dist_temperature = dist_temperature
        self.device = device
        self.image_size = image_size
        self.norm_stats = norm_stats
        self.projection_type = projection_type
        self.tasks = []

        splits_file = os.path.join(data_path, 'splits.csv')
        if not os.path.exists(splits_file):
            logger.warning(f'No splits.csv at {data_path} — ProtoNet evaluator disabled')
            return

        splits_df = pd.read_csv(splits_file, index_col=0)
        val_tasks = sorted(splits_df[splits_df['split'] == 'val']['task'].tolist())

        logger.info(f'Loading {len(val_tasks)} ProtoNet eval tasks from val split...')
        rng = np.random.RandomState(42)

        for tname in val_tasks:
            task_dir = os.path.join(data_path, tname)
            if not os.path.isdir(task_dir):
                continue
            try:
                t = self._load_task(task_dir, tname, rng)
                if t is not None:
                    self.tasks.append(t)
            except Exception as e:
                logger.warning(f'  Skipping eval task {tname}: {e}')

        logger.info(f'ProtoNet evaluator ready: {len(self.tasks)} tasks loaded')

    def _load_task(self, task_dir, name, rng):
        """Load a single task from precomputed tensors with fixed sampling."""
        import pandas as pd

        y_supp = pd.read_csv(os.path.join(task_dir, 'y_supp.csv'), index_col=0)
        y_query = pd.read_csv(os.path.join(task_dir, 'y_query.csv'), index_col=0)

        num_cols = y_supp.select_dtypes(include=[np.number]).columns
        if len(num_cols) == 0:
            return None
        col = num_cols[0]
        ys = y_supp[col]
        yq = y_query[col]

        supp_valid = ~ys.isna()
        query_valid = ~yq.isna()
        ys_np = ys[supp_valid].values.astype(np.float32)
        yq_np = yq[query_valid].values.astype(np.float32)

        if len(ys_np) < 2 or len(yq_np) < 2:
            return None

        # Scale y (fit on support)
        y_scaler = self._StandardScaler()
        ys_np = y_scaler.fit_transform(ys_np.reshape(-1, 1)).flatten()
        yq_np = y_scaler.transform(yq_np.reshape(-1, 1)).flatten()

        # Fixed sampling
        n_spt = min(self.k_spt, len(ys_np))
        n_qry = min(self.k_qry, len(yq_np))
        spt_idx = rng.choice(len(ys_np), n_spt, replace=False)
        qry_idx = rng.choice(len(yq_np), n_qry, replace=False)

        # Load precomputed GADF tensors [N, 1, H, W] float16 → float32
        sx = torch.load(os.path.join(task_dir, 'X_supp.pt'),
                        map_location='cpu').float()[supp_valid.values][spt_idx]
        qx = torch.load(os.path.join(task_dir, 'X_query.pt'),
                        map_location='cpu').float()[query_valid.values][qry_idx]

        if self.norm_stats is not None:
            mean = torch.tensor(self.norm_stats[0]).view(1, 1, 1, 1)
            std = torch.tensor(self.norm_stats[1]).view(1, 1, 1, 1)
            sx = (sx - mean) / std
            qx = (qx - mean) / std

        sy = torch.from_numpy(ys_np[spt_idx]).unsqueeze(1)
        qy = torch.from_numpy(yq_np[qry_idx]).unsqueeze(1)

        return (name, sx, sy, qx, qy)

    def evaluate(self, encoder, projection, device):
        """Run ProtoNet regression evaluation on all val tasks."""
        if not self.tasks:
            return {}

        from sklearn.metrics import r2_score

        # Unwrap DDP
        enc = encoder.module if hasattr(encoder, 'module') else encoder
        enc.eval()
        if projection is not None:
            proj = projection.module if hasattr(projection, 'module') else projection
            proj.eval()
        else:
            proj = None

        r2_list = []
        rmse_list = []

        for name, sx, sy, qx, qy in self.tasks:
            with torch.no_grad():
                supp_emb = enc(sx.to(device))    # [k_spt, N, D]
                query_emb = enc(qx.to(device))   # [k_qry, N, D]

                if proj is not None and self.projection_type == 'transformer':
                    supp_emb = proj(supp_emb)
                    query_emb = proj(query_emb)
                else:
                    supp_emb = supp_emb.mean(dim=1)
                    query_emb = query_emb.mean(dim=1)
                    if proj is not None:
                        supp_emb = proj(supp_emb)
                        query_emb = proj(query_emb)

                # Distance-weighted regression
                dists = -torch.cdist(query_emb, supp_emb) / self.dist_temperature
                weights = torch.softmax(dists, dim=1)
                preds = weights @ sy.to(device)

            preds_np = preds.cpu().numpy().flatten()
            y_true = qy.numpy().flatten()

            r2 = r2_score(y_true, preds_np) if len(y_true) > 1 else 0.0
            mse = np.mean((y_true - preds_np) ** 2)
            rmse = np.sqrt(mse)
            r2_list.append(r2)
            rmse_list.append(rmse)

        return {
            'probe/r2_mean': np.mean(r2_list),
            'probe/r2_std': np.std(r2_list),
            'probe/rmse_mean': np.mean(rmse_list),
            'probe/rmse_std': np.std(rmse_list),
            'probe/n_tasks': len(self.tasks),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- DATA
    dataset_type = args['data'].get('dataset_type', 'imagenet')
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
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

    from datetime import datetime
    if load_model and r_file is not None and os.path.isabs(r_file):
        folder = os.path.dirname(r_file)
    else:
        run_id = args['logging'].get('run_id') or datetime.now().strftime('%Y%m%d_%H%M%S')
        folder = os.path.join(base_folder, f'{tag}_{run_id}')
    os.makedirs(folder, exist_ok=True)

    # -- WANDB
    wandb_cfg = args.get('wandb', {})
    use_wandb = wandb_cfg.get('enable', False)
    wandb_project = wandb_cfg.get('project', 'ijepa-gadf2d-joint')
    wandb_name = wandb_cfg.get('name', None)

    # -- PROBE (ProtoNet evaluation)
    probe_cfg = args.get('probe', {})
    probe_freq = probe_cfg.get('freq', 0)
    probe_data_path = probe_cfg.get('data_path', '')
    probe_k_spt = probe_cfg.get('k_spt', 25)
    probe_k_qry = probe_cfg.get('k_qry', 25)

    # -- PROTONET (joint training branch)
    proto_cfg = args.get('protonet', {})
    proto_enable = proto_cfg.get('enable', False)
    lambda_proto = float(proto_cfg.get('lambda_proto', 1.0))
    proto_data_path = proto_cfg.get('data_path', '')
    proto_k_spt = proto_cfg.get('k_spt', 25)
    proto_k_qry = proto_cfg.get('k_qry', 25)
    proto_dist_temp = float(proto_cfg.get('dist_temperature', 0.5))
    proto_loss_type = str(proto_cfg.get('loss_type', 'mse')).lower()
    proto_use_projection = proto_cfg.get('use_projection', False)
    proto_projection_type = proto_cfg.get('projection_type', 'mlp')  # 'mlp' | 'transformer'
    proto_hidden_dim = proto_cfg.get('projection_hidden_dim', 256)
    proto_output_dim = proto_cfg.get('projection_output_dim', 128)
    aggregator_num_layers = int(proto_cfg.get('aggregator_num_layers', 1))
    aggregator_num_heads = int(proto_cfg.get('aggregator_num_heads', 4))
    aggregator_mlp_ratio = float(proto_cfg.get('aggregator_mlp_ratio', 2.0))
    aggregator_dropout = float(proto_cfg.get('aggregator_dropout', 0.1))
    aggregator_attn_dropout = float(proto_cfg.get('aggregator_attn_dropout', 0.1))
    meta_batch_size = int(proto_cfg.get('meta_batch_size', 8))
    proto_warmup = int(proto_cfg.get('proto_warmup', warmup))  # default: same as I-JEPA warmup

    # -- SIGREG (optional representation regularization)
    sigreg_cfg = args.get('sigreg', {})
    sigreg_enable = bool(sigreg_cfg.get('enable', False))
    sigreg_alpha = float(sigreg_cfg.get('alpha', 0.1))
    sigreg_sketch_dim = int(sigreg_cfg.get('sketch_dim', 64))
    sigreg_warmup = int(sigreg_cfg.get('warmup_epochs', 0))
    sigreg_target = str(sigreg_cfg.get('target', 'jepa_context_blocks'))
    sigreg_representation = str(sigreg_cfg.get('representation', 'mean_tokens'))
    sigreg_grad_clip_norm = sigreg_cfg.get('grad_clip_norm', None)
    sigreg_loss_cap = sigreg_cfg.get('loss_cap', None)
    sigreg_layer_ids = sigreg_cfg.get('layer_ids', None)
    if sigreg_grad_clip_norm is not None:
        sigreg_grad_clip_norm = float(sigreg_grad_clip_norm)
    if sigreg_loss_cap is not None:
        sigreg_loss_cap = float(sigreg_loss_cap)
    if sigreg_enable:
        if sigreg_target != 'jepa_context_blocks':
            raise ValueError(
                f'Invalid sigreg.target={sigreg_target!r}. '
                "Only 'jepa_context_blocks' is supported.")
        if not (0.0 <= sigreg_alpha <= 1.0):
            raise ValueError(f'sigreg.alpha must be in [0, 1], got {sigreg_alpha}')
        if sigreg_sketch_dim <= 0:
            raise ValueError(f'sigreg.sketch_dim must be positive, got {sigreg_sketch_dim}')
        if sigreg_representation not in {'mean_tokens', 'tokens'}:
            raise ValueError(
                f'Invalid sigreg.representation={sigreg_representation!r}. '
                "Expected 'mean_tokens' or 'tokens'.")

    valid_proto_losses = {'mse', 'smooth_l1'}
    if proto_loss_type not in valid_proto_losses:
        raise ValueError(
            f'Invalid protonet.loss_type={proto_loss_type!r}. '
            f'Expected one of {sorted(valid_proto_losses)}'
        )

    dump = os.path.join(folder, 'params-joint.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')
    if rank > 0:
        logger.setLevel(logging.ERROR)

    # -- W&B (rank 0 only)
    if use_wandb and rank == 0:
        try:
            import wandb
            if wandb_name is None:
                wandb_name = f'{tag}_{model_name}_joint_ps{patch_size}_bs{batch_size}'
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
        else:
            load_path = latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'loss_jepa'),
                           ('%.5f', 'loss_proto'),
                           ('%.5f', 'loss_sigreg'),
                           ('%.5f', 'alpha_sigreg'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    in_chans = 1 if dataset_type == 'gadf' else 3
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name,
        in_chans=in_chans)
    target_encoder = copy.deepcopy(encoder)

    # -- init ProtoNet projection (optional)
    proto_projection = None
    if proto_enable and proto_use_projection:
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
        logger.info(f'ProtoNet projection ({proto_projection_type}): {sum(p.numel() for p in proto_projection.parameters()):,} params')

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep,
        symmetric_masking=(dataset_type == 'gadf'))

    # -- Load GADF norm stats
    gadf_norm_stats_path = args['data'].get('gadf_norm_stats', None)
    use_diagonal = args['data'].get('gadf_global_stats', None) is not None
    if dataset_type == 'gadf' and gadf_norm_stats_path is not None:
        from src.gadf_utils import load_norm_stats
        norm_stats_tuple = load_norm_stats(gadf_norm_stats_path,
                                           use_diagonal=use_diagonal)
        logger.info(f'GADF norm stats from {gadf_norm_stats_path} '
                    f'(diagonal={use_diagonal}): '
                    f'mean={norm_stats_tuple[0]}, std={norm_stats_tuple[1]}')
    elif dataset_type == 'gadf':
        norm_stats_tuple = ((-0.0000,), (0.5922,))
        logger.warning('No gadf_norm_stats in config, using hardcoded defaults')
    else:
        norm_stats_tuple = None

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter,
        in_chans=in_chans,
        norm_stats=norm_stats_tuple)

    # -- init data-loaders/samplers
    assert dataset_type == 'gadf', 'Joint training only supports GADF dataset'
    _, unsupervised_loader, unsupervised_sampler = make_gadf(
        transform=transform,
        batch_size=batch_size,
        collator=mask_collator,
        pin_mem=pin_mem,
        num_workers=num_workers,
        world_size=world_size,
        rank=rank,
        h5_path=args['data']['gadf_h5_path'],
        drop_last=True)
    ipe = len(unsupervised_loader)

    # -- Diagonal encoding stats for episodic/eval tasks
    gadf_global_stats_path = args['data'].get('gadf_global_stats', None)
    task_global_stats = None
    if gadf_global_stats_path is not None:
        from src.gadf_utils import load_global_stats
        task_global_stats = load_global_stats(gadf_global_stats_path)
        logger.info(f'Diagonal encoding for tasks: min={task_global_stats[0]:.4f}, '
                    f'max={task_global_stats[1]:.4f}')

    # -- Episodic task sampler (rank 0 for single-GPU; all ranks sample independently)
    proto_sampler = None
    if proto_enable and dataset_type == 'gadf':
        proto_sampler = EpisodicTaskSampler(
            data_path=proto_data_path,
            image_size=crop_size,
            norm_stats=norm_stats_tuple,
            k_spt=proto_k_spt,
            k_qry=proto_k_qry,
            global_stats=task_global_stats,
        )
        if not proto_sampler.tasks:
            logger.warning('No episodic tasks loaded — disabling ProtoNet branch')
            proto_sampler = None

    # -- ProtoNet evaluator (rank 0 only)
    proto_evaluator = None
    if probe_freq > 0 and rank == 0 and dataset_type == 'gadf':
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

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)

    # -- Add projection params to optimizer (if enabled)
    if proto_projection is not None:
        no_wd_names = set()
        if proto_projection_type == 'transformer':
            # Keep transformer readout tokens/position embeddings out of weight
            # decay. Unlike MLP weights, these special parameters act as learned
            # references and are easy to over-regularize with the JEPA schedule.
            no_wd_names.update({'cls_token', 'pos_embed'})

        optimizer.add_param_group({
            'params': [p for n, p in proto_projection.named_parameters()
                       if ('bias' not in n)
                       and (len(p.shape) != 1)
                       and (n not in no_wd_names)]
        })
        optimizer.add_param_group({
            'params': [p for n, p in proto_projection.named_parameters()
                       if ('bias' in n)
                       or (len(p.shape) == 1)
                       or (n in no_wd_names)],
            'WD_exclude': True,
            'weight_decay': 0
        })

    # -- DDP wrapping
    # When ProtoNet is enabled, the encoder is called in two modes per epoch:
    # (1) masked forward during JEPA iterations, and
    # (2) unmasked forward during the epoch-level ProtoNet step.
    # Disable static_graph for encoder to handle the graph change between phases.
    encoder_static = not (proto_sampler is not None)
    if world_size > 1:
        encoder = DistributedDataParallel(encoder, static_graph=encoder_static)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
        if proto_projection is not None:
            proto_projection = DistributedDataParallel(proto_projection, static_graph=True)
    for p in target_encoder.parameters():
        p.requires_grad = False

    sigreg_collector = None
    if sigreg_enable:
        sigreg_collector = SigRegHookCollector(
            encoder,
            sketch_dim=sigreg_sketch_dim,
            representation=sigreg_representation,
            layer_ids=sigreg_layer_ids,
        )
        logger.info(
            'Weak-SIGReg enabled: target=%s representation=%s alpha=%.4f '
            'sketch_dim=%d warmup_epochs=%d grad_clip_norm=%s loss_cap=%s'
            % (sigreg_target, sigreg_representation, sigreg_alpha,
               sigreg_sketch_dim, sigreg_warmup, sigreg_grad_clip_norm,
               sigreg_loss_cap)
        )

    # -- momentum schedule
    # Accounts for interleaved ProtoNet EMA steps: ceil(n_tasks / meta_batch_size) per epoch
    n_proto_steps_per_epoch = math.ceil(len(proto_sampler) / meta_batch_size) if proto_sampler is not None else 0
    _proto_ema_steps = n_proto_steps_per_epoch * num_epochs
    _total_ema_steps = int(ipe*num_epochs*ipe_scale) + _proto_ema_steps
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/_total_ema_steps
                          for i in range(_total_ema_steps + 1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        # Load projection state if present
        if proto_projection is not None and load_path is not None:
            try:
                ckpt = torch.load(load_path, map_location='cpu')
                if 'proto_projection' in ckpt:
                    proto_projection.load_state_dict(ckpt['proto_projection'])
                    logger.info('Loaded proto_projection from checkpoint')
                else:
                    logger.info('No proto_projection in checkpoint — starting fresh')
                del ckpt
            except Exception as e:
                logger.warning(f'Could not load proto_projection: {e}')
        for _ep in range(start_epoch):
            for _ in range(ipe):
                scheduler.step()
                wd_scheduler.step()
                next(momentum_scheduler)
                mask_collator.step()
            for _ in range(n_proto_steps_per_epoch):
                next(momentum_scheduler)  # one EMA step per interleaved ProtoNet step

    best_probe_r2 = -float('inf')
    best_ckpt_artifact = None
    last_logged_artifact = None

    def save_checkpoint(epoch, lambda_eff=0.0):
        nonlocal last_logged_artifact
        _loss_proto_avg = loss_proto_meter.avg
        _loss_sigreg_avg = loss_sigreg_meter.avg
        _loss_jepa_step_avg = loss_jepa_step_meter.avg
        _loss_total = _loss_jepa_step_avg + lambda_eff * _loss_proto_avg
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss_jepa': loss_jepa_meter.avg,
            'loss_proto': _loss_proto_avg,
            'loss_sigreg': _loss_sigreg_avg,
            'loss_jepa_step': _loss_jepa_step_avg,
            'loss_total': _loss_total,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
            'sigreg': sigreg_cfg,
        }
        if proto_projection is not None:
            save_dict['proto_projection'] = proto_projection.state_dict()
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                ckpt_path = save_path.format(epoch=f'{epoch + 1}')
                torch.save(save_dict, ckpt_path)
                if use_wandb:
                    import wandb
                    try:
                        art = wandb.Artifact(
                            f'{tag}-checkpoint',
                            type='model',
                            metadata={'epoch': epoch + 1,
                                      'loss_jepa': loss_jepa_meter.avg,
                                      'loss_proto': _loss_proto_avg,
                                      'loss_sigreg': _loss_sigreg_avg,
                                      'loss_jepa_step': _loss_jepa_step_avg,
                                      'loss_total': _loss_total},
                        )
                        art.ttl = artifact_ttl
                        art.add_reference(f'file://{os.path.abspath(ckpt_path)}')
                        logged = wandb.log_artifact(art)
                        logged.wait()
                        last_logged_artifact = logged.name
                    except OSError as e:
                        logger.warning(f'[wandb artifact] add_reference failed ({e}), retrying without file reference')
                        try:
                            art = wandb.Artifact(
                                f'{tag}-checkpoint',
                                type='model',
                                metadata={'epoch': epoch + 1,
                                          'loss_jepa': loss_jepa_meter.avg,
                                          'loss_proto': _loss_proto_avg,
                                          'loss_sigreg': _loss_sigreg_avg,
                                          'loss_jepa_step': _loss_jepa_step_avg,
                                          'loss_total': _loss_total,
                                          'ckpt_path': os.path.abspath(ckpt_path)},
                            )
                            logged = wandb.log_artifact(art)
                            logged.wait()
                            last_logged_artifact = logged.name
                        except Exception as e2:
                            logger.warning(f'[wandb artifact] Skipped artifact logging entirely: {e2}')

    # -- Proto mini-batch step helper (captures encoder/optimizer/scaler/etc. from scope)
    def proto_minibatch_step(episodes, lambda_eff):
        """Gradient-accumulate over a mini-batch of episodes, then do one optimizer step + EMA."""
        n_tasks = len(episodes)
        loss_accum = 0.0
        for (spt_imgs, spt_y, qry_imgs, qry_y) in episodes:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                spt_emb = encoder(spt_imgs)   # [k_spt, N, D]
                qry_emb = encoder(qry_imgs)   # [k_qry, N, D]
                if proto_projection is not None and proto_projection_type == 'transformer':
                    # TransformerAggregator handles aggregation internally
                    spt_emb = proto_projection(spt_emb)   # [k_spt, output_dim]
                    qry_emb = proto_projection(qry_emb)   # [k_qry, output_dim]
                else:
                    spt_emb = spt_emb.mean(dim=1)         # [k_spt, D]
                    qry_emb = qry_emb.mean(dim=1)         # [k_qry, D]
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
                # Scale so accumulated grads = mean over tasks in this mini-batch
                task_loss_scaled = lambda_eff * task_loss / n_tasks
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

        # EMA update for this ProtoNet step
        with torch.no_grad():
            m = next(momentum_scheduler)
            for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

        return loss_accum / n_tasks

    def _clip_optimizer_grad_norm(max_norm):
        params = [
            p
            for group in optimizer.param_groups
            for p in group['params']
            if p.grad is not None
        ]
        if not params:
            return 0.0
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm))

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        unsupervised_sampler.set_epoch(epoch)

        loss_jepa_meter = AverageMeter()
        loss_jepa_step_meter = AverageMeter()
        loss_proto_meter = AverageMeter()
        loss_sigreg_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        # -- ProtoNet interleaving setup for this epoch
        if proto_sampler is not None:
            task_order = np.random.permutation(len(proto_sampler))
            _n_proto_steps = math.ceil(len(task_order) / meta_batch_size)
            proto_freq = max(1, ipe // _n_proto_steps)
            task_ptr = 0
            # Lambda warmup: ramp from 0 to lambda_proto over proto_warmup epochs
            lambda_eff = lambda_proto * min(1.0, (epoch + 1) / proto_warmup) if proto_warmup > 0 else lambda_proto
            logger.info('Epoch %d ProtoNet: %d tasks, %d steps, freq=%d, lambda_eff=%.5f'
                        % (epoch + 1, len(task_order), _n_proto_steps, proto_freq, lambda_eff))
        else:
            lambda_eff = lambda_proto

        if sigreg_enable:
            alpha_sigreg_eff = (
                sigreg_alpha * min(1.0, (epoch + 1) / sigreg_warmup)
                if sigreg_warmup > 0 else sigreg_alpha
            )
        else:
            alpha_sigreg_eff = 0.0

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                # ============================================================
                # I-JEPA branch (identical to src/train.py)
                # ============================================================

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))
                        B = len(h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    if sigreg_collector is not None and alpha_sigreg_eff > 0.0:
                        with sigreg_collector.capture():
                            z = encoder(imgs, masks_enc)
                        loss_sigreg = sigreg_collector.loss()
                        if loss_sigreg is None:
                            loss_sigreg = z.sum() * 0.0
                    else:
                        z = encoder(imgs, masks_enc)
                        loss_sigreg = z.sum() * 0.0
                    z = predictor(z, masks_enc, masks_pred)
                    return z, loss_sigreg

                def jepa_loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z, loss_sigreg = forward_context()
                    loss_jepa = jepa_loss_fn(z, h)
                    loss_sigreg = AllReduce.apply(loss_sigreg)
                    loss_sigreg_for_bp = (
                        torch.clamp(loss_sigreg, max=sigreg_loss_cap)
                        if sigreg_loss_cap is not None else loss_sigreg
                    )
                    loss_jepa_step = (
                        (1.0 - alpha_sigreg_eff) * loss_jepa
                        + alpha_sigreg_eff * loss_sigreg_for_bp
                    )

                # Backward & step
                if use_bfloat16:
                    scaler.scale(loss_jepa_step).backward()
                    if sigreg_grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        _clip_optimizer_grad_norm(sigreg_grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_jepa_step.backward()
                    if sigreg_grad_clip_norm is not None:
                        _clip_optimizer_grad_norm(sigreg_grad_clip_norm)
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # EMA update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (
                    float(loss_jepa),
                    float(loss_sigreg),
                    float(loss_jepa_step),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                )

            result, etime = gpu_timer(train_step)
            (loss_jepa_val, loss_sigreg_val, loss_jepa_step_val,
             _new_lr, _new_wd, grad_stats) = result
            loss_jepa_meter.update(loss_jepa_val)
            loss_sigreg_meter.update(loss_sigreg_val)
            loss_jepa_step_meter.update(loss_jepa_step_val)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss_jepa_step_val, loss_jepa_val,
                               loss_proto_meter.avg, loss_sigreg_val, alpha_sigreg_eff,
                               maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss_jepa_step_val) or np.isinf(loss_jepa_step_val):
                    logger.info('[%d, %5d] loss: %.3f loss_jepa: %.3f '
                                'loss_sigreg: %.4f alpha_sigreg: %.4f '
                                'loss_proto: %.4f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_jepa_step_meter.avg,
                                   loss_jepa_meter.avg,
                                   loss_sigreg_meter.avg,
                                   alpha_sigreg_eff,
                                   loss_proto_meter.avg,
                                   maskA_meter.avg,
                                   maskB_meter.avg,
                                   _new_wd,
                                   _new_lr,
                                   torch.cuda.max_memory_allocated() / 1024.**2,
                                   time_meter.avg))

                    if grad_stats is not None:
                        logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                    % (epoch + 1, itr,
                                       grad_stats.first_layer,
                                       grad_stats.last_layer,
                                       grad_stats.min,
                                       grad_stats.max))

            log_stats()

            assert not np.isnan(loss_jepa_step_val), 'loss is nan'

            # ================================================================
            # Interleaved ProtoNet step (every proto_freq I-JEPA iterations)
            # ================================================================
            if proto_sampler is not None and (itr + 1) % proto_freq == 0 and task_ptr < len(task_order):
                batch_end = min(task_ptr + meta_batch_size, len(task_order))
                batch_idx = task_order[task_ptr:batch_end]
                episodes = [proto_sampler.sample_episode_by_idx(int(i), device) for i in batch_idx]
                loss_proto_val = proto_minibatch_step(episodes, lambda_eff)
                loss_proto_meter.update(loss_proto_val)
                task_ptr = batch_end

        # -- Handle remaining tasks if I-JEPA loop ended before all tasks were consumed
        if proto_sampler is not None:
            while task_ptr < len(task_order):
                batch_end = min(task_ptr + meta_batch_size, len(task_order))
                batch_idx = task_order[task_ptr:batch_end]
                episodes = [proto_sampler.sample_episode_by_idx(int(i), device) for i in batch_idx]
                loss_proto_val = proto_minibatch_step(episodes, lambda_eff)
                loss_proto_meter.update(loss_proto_val)
                task_ptr = batch_end

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f  loss_jepa %.3f  loss_sigreg %.4f  '
                    'loss_proto %.4f  (alpha_sigreg=%.5f, lambda_eff=%.5f, %d proto steps)'
                    % (loss_jepa_step_meter.avg, loss_jepa_meter.avg,
                       loss_sigreg_meter.avg, loss_proto_meter.avg,
                       alpha_sigreg_eff, lambda_eff,
                       _n_proto_steps if proto_sampler is not None else 0))
        save_checkpoint(epoch+1, lambda_eff=lambda_eff)

        # -- W&B + ProtoNet eval logging (rank 0)
        if rank == 0:
            wandb_log = {
                'train/loss_jepa': loss_jepa_meter.avg,
                'train/loss_jepa_step': loss_jepa_step_meter.avg,
                'train/loss_sigreg': loss_sigreg_meter.avg,
                'train/loss_proto': loss_proto_meter.avg,
                'train/loss_total': loss_jepa_step_meter.avg + lambda_eff * loss_proto_meter.avg,
                'train/lambda_proto': lambda_eff,
                'train/alpha_sigreg': alpha_sigreg_eff,
                'train/mask_a': maskA_meter.avg,
                'train/mask_b': maskB_meter.avg,
                'train/lr': _new_lr,
                'train/wd': _new_wd,
                'train/time_ms': time_meter.avg,
                'epoch': epoch + 1,
            }

            # -- Online ProtoNet evaluation
            if proto_evaluator is not None and (epoch + 1) % probe_freq == 0:
                t0 = time.time()
                probe_metrics = proto_evaluator.evaluate(
                    target_encoder, proto_projection, device)
                probe_time = time.time() - t0
                logger.info(
                    f'ProtoNet eval (ep {epoch+1}): '
                    f'R²={probe_metrics["probe/r2_mean"]:.4f}±'
                    f'{probe_metrics["probe/r2_std"]:.4f}  '
                    f'RMSE={probe_metrics["probe/rmse_mean"]:.4f}±'
                    f'{probe_metrics["probe/rmse_std"]:.4f}  '
                    f'({probe_metrics["probe/n_tasks"]} tasks, {probe_time:.1f}s)'
                )
                wandb_log.update(probe_metrics)
                # Track best probe R²
                r2 = probe_metrics['probe/r2_mean']
                if r2 > best_probe_r2 and last_logged_artifact is not None:
                    best_probe_r2 = r2
                    best_ckpt_artifact = last_logged_artifact
                    logger.info(f'New best probe R²={r2:.4f} at epoch {epoch+1}'
                                f' → artifact {best_ckpt_artifact}')
                # Restore training mode
                encoder.train()
                predictor.train()
                if proto_projection is not None:
                    proto_projection.train()

            if use_wandb:
                import wandb
                wandb.log(wandb_log)

    # -- Log final checkpoint as artifact (no TTL) and remove TTL from best
    if use_wandb and rank == 0:
        import wandb
        art = wandb.Artifact(
            f'{tag}-checkpoint-latest',
            type='model',
            metadata={
                'epoch': num_epochs,
                'loss_jepa': loss_jepa_meter.avg,
                'loss_sigreg': loss_sigreg_meter.avg,
                'loss_jepa_step': loss_jepa_step_meter.avg,
                'loss_proto': loss_proto_meter.avg,
                'loss_total': loss_jepa_step_meter.avg + lambda_eff * loss_proto_meter.avg,
            },
        )
        art.add_reference(f'file://{os.path.abspath(latest_path)}')
        wandb.log_artifact(art)

        if best_ckpt_artifact is not None:
            try:
                api = wandb.Api()
                full_name = f'{wandb.run.entity}/{wandb.run.project}/{best_ckpt_artifact}'
                best_art = api.artifact(full_name)
                best_art.ttl = None
                best_art.save()
                logger.info(f'Removed TTL from best artifact: {full_name}')
            except Exception as e:
                logger.warning(f'Could not remove TTL from best artifact: {e}')

    # -- Finish W&B
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
