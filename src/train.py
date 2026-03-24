# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import logging
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
from src.datasets.imagenet1k import make_imagenet1k
from src.datasets.gadf_dataset import make_gadf

from src.helper import (
    load_checkpoint,
    init_model,
    init_opt)
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
# Online linear probe evaluator (GADF 2D)
# ---------------------------------------------------------------------------

class LinearProbeEvaluator:
    """Pre-loads val-split tasks and runs quick linear probes on frozen encoder.

    All tasks from the 'val' split of splits.csv are loaded once at init.
    Spectra are converted to GADF 2D images once and cached in CPU memory.
    Support/query samples are drawn with a fixed seed so that every call to
    ``evaluate()`` uses the exact same data.
    """

    def __init__(self, data_path, image_size=224, norm_stats=None,
                 k_spt=25, k_qry=25, probe_epochs=50, device='cpu',
                 global_stats=None):
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        from pyts.image import GramianAngularField
        self._StandardScaler = StandardScaler

        self.probe_epochs = probe_epochs
        self.device = device
        self.image_size = image_size
        self.norm_stats = norm_stats
        self.global_stats = global_stats  # (global_min, global_max) or None
        self.tasks = []  # list of (name, supp_x, supp_y, query_x, query_y)

        splits_file = os.path.join(data_path, 'splits.csv')
        if not os.path.exists(splits_file):
            logger.warning(f'No splits.csv at {data_path} — probe disabled')
            return

        splits_df = pd.read_csv(splits_file, index_col=0)
        val_tasks = sorted(splits_df[splits_df['split'] == 'val']['task'].tolist())

        logger.info(f'Loading {len(val_tasks)} probe tasks from val split...')
        rng = np.random.RandomState(42)  # fixed seed — same across experiments
        gaf = GramianAngularField(image_size=image_size, method='difference')

        for tname in val_tasks:
            task_dir = os.path.join(data_path, tname)
            if not os.path.isdir(task_dir):
                continue
            try:
                t = self._load_task(task_dir, tname, k_spt, k_qry, rng, gaf)
                if t is not None:
                    self.tasks.append(t)
            except Exception as e:
                logger.warning(f'  Skipping probe task {tname}: {e}')

        logger.info(f'Probe evaluator ready: {len(self.tasks)} tasks loaded')

    def _load_task(self, task_dir, name, k_spt, k_qry, rng, gaf):
        """Load a single task: read CSV, convert to GADF, scale y, sample."""
        import pandas as pd

        X_supp = pd.read_csv(os.path.join(task_dir, 'X_supp.csv'), index_col=0)
        y_supp = pd.read_csv(os.path.join(task_dir, 'y_supp.csv'), index_col=0)
        X_query = pd.read_csv(os.path.join(task_dir, 'X_query.csv'), index_col=0)
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
        X_supp = X_supp[supp_valid].values.astype(np.float32)
        ys = ys[supp_valid].values.astype(np.float32)
        X_query = X_query[query_valid].values.astype(np.float32)
        yq = yq[query_valid].values.astype(np.float32)

        if len(ys) < 2 or len(yq) < 2:
            return None

        # Scale y (fit on support)
        y_scaler = self._StandardScaler()
        ys = y_scaler.fit_transform(ys.reshape(-1, 1)).flatten()
        yq = y_scaler.transform(yq.reshape(-1, 1)).flatten()

        # Sample fixed support/query subsets (deterministic via rng)
        n_spt = min(k_spt, len(ys))
        n_qry = min(k_qry, len(yq))
        spt_idx = rng.choice(len(ys), n_spt, replace=False)
        qry_idx = rng.choice(len(yq), n_qry, replace=False)

        # Convert sampled spectra → GADF 2D images [N, 1, H, W]
        supp_gadf = gaf.transform(X_supp[spt_idx])  # [n_spt, H, W]
        query_gadf = gaf.transform(X_query[qry_idx])  # [n_qry, H, W]

        if self.global_stats is not None:
            from src.gadf_utils import encode_diagonal
            encode_diagonal(supp_gadf, X_supp[spt_idx],
                            *self.global_stats, self.image_size)
            encode_diagonal(query_gadf, X_query[qry_idx],
                            *self.global_stats, self.image_size)

        sx = torch.from_numpy(supp_gadf.astype(np.float32)).unsqueeze(1)
        qx = torch.from_numpy(query_gadf.astype(np.float32)).unsqueeze(1)

        # Normalize GADF (same stats as pretraining)
        if self.norm_stats is not None:
            mean = torch.tensor(self.norm_stats[0]).view(1, 1, 1, 1)
            std = torch.tensor(self.norm_stats[1]).view(1, 1, 1, 1)
            sx = (sx - mean) / std
            qx = (qx - mean) / std

        sy = torch.from_numpy(ys[spt_idx]).unsqueeze(1)
        qy = torch.from_numpy(yq[qry_idx]).unsqueeze(1)

        return (name, sx, sy, qx, qy)

    def evaluate(self, target_encoder, device):
        """Run linear probe on all tasks. Returns dict of mean metrics."""
        if not self.tasks:
            return {}

        from sklearn.metrics import r2_score

        # Unwrap DDP if needed
        encoder = target_encoder.module if hasattr(target_encoder, 'module') else target_encoder
        encoder.eval()

        r2_list = []
        rmse_list = []

        for name, sx, sy, qx, qy in self.tasks:
            # Extract features with frozen encoder (no grad)
            with torch.no_grad():
                supp_feats = encoder(sx.to(device)).mean(dim=1).detach()
                query_feats = encoder(qx.to(device)).mean(dim=1).detach()
            supp_y = sy.to(device)
            query_y = qy.to(device)

            # Train linear head
            D = supp_feats.shape[1]
            head = nn.Linear(D, 1).to(device)
            opt = torch.optim.Adam(head.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()

            best_val_loss = float('inf')
            patience_counter = 0
            best_pred = None

            for _ in range(self.probe_epochs):
                head.train()
                pred = head(supp_feats)
                loss = loss_fn(pred, supp_y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                head.eval()
                with torch.no_grad():
                    val_pred = head(query_feats)
                    val_loss = loss_fn(val_pred, query_y).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_pred = val_pred.cpu().numpy().flatten()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        break

            # Metrics
            y_true = query_y.cpu().numpy().flatten()
            r2 = r2_score(y_true, best_pred)
            rmse = np.sqrt(best_val_loss)
            r2_list.append(r2)
            rmse_list.append(rmse)

        return {
            'probe/r2_mean': np.mean(r2_list),
            'probe/r2_std': np.std(r2_list),
            'probe/rmse_mean': np.mean(rmse_list),
            'probe/rmse_std': np.std(rmse_list),
            'probe/n_tasks': len(self.tasks),
        }


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
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']  # whether to allow overlap b/w context and target blocks
    patch_size = args['mask']['patch_size']  # patch-size for model training
    num_enc_masks = args['mask']['num_enc_masks']  # number of context blocks
    min_keep = args['mask']['min_keep']  # min number of patches in context block
    enc_mask_scale = args['mask']['enc_mask_scale']  # scale of context blocks
    num_pred_masks = args['mask']['num_pred_masks']  # number of target blocks
    pred_mask_scale = args['mask']['pred_mask_scale']  # scale of target blocks
    aspect_ratio = args['mask']['aspect_ratio']  # aspect ratio of target blocks
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']  # scheduler scale factor (def: 1.0)
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

    # Auto-create timestamped subfolder: logs/gadf_soil_nir/gadf_jepa_20260311_143022/
    # When resuming, reuse the folder from read_checkpoint if it's a full path
    from datetime import datetime
    if load_model and r_file is not None and os.path.isabs(r_file):
        # Absolute checkpoint path → save new outputs next to it
        folder = os.path.dirname(r_file)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        folder = os.path.join(base_folder, f'{tag}_{timestamp}')
    os.makedirs(folder, exist_ok=True)

    # -- WANDB
    wandb_cfg = args.get('wandb', {})
    use_wandb = wandb_cfg.get('enable', False)
    wandb_project = wandb_cfg.get('project', 'ijepa-gadf2d')
    wandb_name = wandb_cfg.get('name', None)

    # -- PROBE
    probe_cfg = args.get('probe', {})
    probe_freq = probe_cfg.get('freq', 0)
    probe_data_path = probe_cfg.get('data_path', '')
    probe_k_spt = probe_cfg.get('k_spt', 25)
    probe_k_qry = probe_cfg.get('k_qry', 25)
    probe_epochs = probe_cfg.get('epochs', 50)

    dump = os.path.join(folder, 'params-ijepa.yaml')
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
                wandb_name = f'{tag}_{model_name}_ps{patch_size}_bs{batch_size}'
            wandb.init(project=wandb_project, name=wandb_name, config=args)
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
            # Support absolute paths and paths relative to base_folder
            load_path = r_file if os.path.isabs(r_file) else os.path.join(base_folder, r_file)
        else:
            load_path = latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
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

    # -- Load GADF norm stats from file (or fallback to defaults)
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
    if dataset_type == 'gadf':
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
    else:
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=world_size,
            rank=rank,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(unsupervised_loader)

    # -- Online linear probe evaluator (rank 0 only)
    probe_evaluator = None
    if probe_freq > 0 and rank == 0 and dataset_type == 'gadf':
        norm_stats_probe = norm_stats_tuple  # same norm as pretraining transforms
        # Diagonal encoding: cargar stats globales si se especificó en config
        gadf_global_stats_path = args['data'].get('gadf_global_stats', None)
        probe_global_stats = None
        if gadf_global_stats_path is not None:
            from src.gadf_utils import load_global_stats
            probe_global_stats = load_global_stats(gadf_global_stats_path)
            logger.info(f'Diagonal encoding for probe: min={probe_global_stats[0]:.4f}, '
                        f'max={probe_global_stats[1]:.4f}')
        probe_evaluator = LinearProbeEvaluator(
            data_path=probe_data_path,
            image_size=crop_size,
            norm_stats=norm_stats_probe,
            k_spt=probe_k_spt,
            k_qry=probe_k_qry,
            probe_epochs=probe_epochs,
            device=device,
            global_stats=probe_global_stats,
        )
        if not probe_evaluator.tasks:
            probe_evaluator = None

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
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

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
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    best_probe_r2 = -float('inf')
    best_ckpt_artifact = None
    last_logged_artifact = None  # most recent periodic artifact version name

    def save_checkpoint(epoch):
        nonlocal last_logged_artifact
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)
            if (epoch + 1) % checkpoint_freq == 0:
                ckpt_path = save_path.format(epoch=f'{epoch + 1}')
                torch.save(save_dict, ckpt_path)
                # Log as wandb artifact with TTL
                if use_wandb:
                    import wandb
                    art = wandb.Artifact(
                        f'{tag}-checkpoint',
                        type='model',
                        metadata={'epoch': epoch + 1, 'loss': loss_meter.avg},
                        ttl=artifact_ttl,
                    )
                    art.add_file(ckpt_path)
                    logged = wandb.log_artifact(art)
                    logged.wait()
                    last_logged_artifact = logged.name

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                # -- unsupervised imgs
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
                # --

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
                        B = len(h)
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    loss = AllReduce.apply(loss)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                #  Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (float(loss), _new_lr, _new_wd, grad_stats)
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info('[%d, %5d] loss: %.3f '
                                'masks: %.1f %.1f '
                                '[wd: %.2e] [lr: %.2e] '
                                '[mem: %.2e] '
                                '(%.1f ms)'
                                % (epoch + 1, itr,
                                   loss_meter.avg,
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

            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint after every epoch
        logger.info('avg. loss %.3f' % loss_meter.avg)
        save_checkpoint(epoch+1)

        # -- W&B + probe logging (rank 0)
        if rank == 0:
            wandb_log = {
                'train/loss': loss_meter.avg,
                'train/mask_a': maskA_meter.avg,
                'train/mask_b': maskB_meter.avg,
                'train/lr': _new_lr,
                'train/wd': _new_wd,
                'train/time_ms': time_meter.avg,
                'epoch': epoch + 1,
            }

            # -- Online linear probe
            if probe_evaluator is not None and (epoch + 1) % probe_freq == 0:
                t0 = time.time()
                probe_metrics = probe_evaluator.evaluate(target_encoder, device)
                probe_time = time.time() - t0
                logger.info(
                    f'Probe (ep {epoch+1}): '
                    f'R²={probe_metrics["probe/r2_mean"]:.4f}±'
                    f'{probe_metrics["probe/r2_std"]:.4f}  '
                    f'RMSE={probe_metrics["probe/rmse_mean"]:.4f}±'
                    f'{probe_metrics["probe/rmse_std"]:.4f}  '
                    f'({probe_metrics["probe/n_tasks"]} tasks, {probe_time:.1f}s)'
                )
                wandb_log.update(probe_metrics)
                # Track best probe R² for artifact TTL
                r2 = probe_metrics['probe/r2_mean']
                if r2 > best_probe_r2 and last_logged_artifact is not None:
                    best_probe_r2 = r2
                    best_ckpt_artifact = last_logged_artifact
                    logger.info(f'New best probe R²={r2:.4f} at epoch {epoch+1}'
                                f' → artifact {best_ckpt_artifact}')
                # Restore training mode
                encoder.train()
                predictor.train()

            if use_wandb:
                import wandb
                wandb.log(wandb_log)

    # -- Log final checkpoint as artifact (no TTL) and remove TTL from best
    if use_wandb and rank == 0:
        import wandb
        # Log latest checkpoint without TTL
        art = wandb.Artifact(
            f'{tag}-checkpoint-latest',
            type='model',
            metadata={'epoch': num_epochs, 'loss': loss_meter.avg},
        )
        art.add_file(latest_path)
        wandb.log_artifact(art)

        # Remove TTL from best checkpoint artifact
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
