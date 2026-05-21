# ijepa_spectra2d — Downstream fine-tuning (regression) con imágenes GADF 2D
#
# Carga el target_encoder de un checkpoint de pretraining I-JEPA 2D y añade
# una cabeza de regresión.  Los datos downstream (X_supp.csv / X_query.csv)
# contienen espectros NIR que se convierten a imágenes GADF on-the-fly.
#
# Modos de dataset:
#   --region_tasks   Cada subdirectorio del data_path es una tarea
#   --simple_dataset Un solo directorio con y_supp.csv multi-columna
#
# Modos de entrenamiento:
#   (default)         Full fine-tuning
#   --linear_probing  Congela el encoder, solo entrena la cabeza
#
# Ejemplo:
#   python main_finetune_gadf2d.py \
#       --data_path ../SpectraI-JEPA/data/Soil_NIR_AGG_mixed \
#       --region_tasks --split test \
#       --model_weight logs/gadf_soil_nir/gadf_jepa-latest.pth.tar \
#       --save_path results/ft_gadf2d/ \
#       --device cuda:0

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

# Local imports
import src.models.vision_transformer as vit

# GADF conversion
from pyts.image import GramianAngularField

# Target scaling
from sklearn.preprocessing import StandardScaler


# ============================================================================
# EarlyStop
# ============================================================================

class EarlyStop:
    """Para el entrenamiento si la métrica no mejora durante `patience` épocas."""

    def __init__(self, patience=20, mode='min'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, model, path, epoch):
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == 'min' and score < self.best_score:
            improved = True
        elif self.mode == 'max' and score > self.best_score:
            improved = True

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ============================================================================
# Pooling modules
# ============================================================================

class AvgPool(nn.Module):
    """Global average pooling sobre patch tokens."""
    def forward(self, x):  # [B, N, D]
        return x.mean(dim=1)  # [B, D]


class AttentionPool(nn.Module):
    """Attention pooling: score independiente por patch, weighted sum."""
    def __init__(self, embed_dim):
        super().__init__()
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, x):  # [B, N, D]
        w = self.attn(x).softmax(dim=1)  # [B, N, 1]
        return (w * x).sum(dim=1)         # [B, D]


class MultiHeadAttentionPool(nn.Module):
    """MAP: cross-attention con query learnable sobre patch tokens."""
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):  # [B, N, D]
        q = self.query.expand(x.size(0), -1, -1)  # [B, 1, D]
        out, _ = self.attn(q, x, x)               # [B, 1, D]
        return out.squeeze(1)                       # [B, D]


def make_pool(mode, embed_dim):
    """Factory para crear el módulo de pooling."""
    if mode == 'avg':
        return AvgPool()
    elif mode == 'attn':
        return AttentionPool(embed_dim)
    elif mode == 'map':
        return MultiHeadAttentionPool(embed_dim)
    else:
        raise ValueError(f"Unknown pool_mode: {mode}")


# ============================================================================
# GADF2DForRegression
# ============================================================================

class GADF2DForRegression(nn.Module):
    """Cabeza de regresión sobre el ViT encoder pretrained (2D GADF).

    Forward:
        x [B, 1, 224, 224] → encoder → [B, N, D] → pool → [B, D]
                           → head → [B, out_dim]
    """

    def __init__(self, encoder: nn.Module, embed_dim: int, out_dim: int = 1,
                 pool=None):
        super().__init__()
        self.encoder = encoder
        self.pool = pool or AvgPool()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x):
        z = self.encoder(x, masks=None)   # [B, N, D]
        z = self.pool(z)                  # [B, D]
        return self.head(z)               # [B, out_dim]


# ============================================================================
# HeadOnlyModel — para linear probing con features pre-extraídas
# ============================================================================

class HeadOnlyModel(nn.Module):
    """Aplica pooling + cabeza de regresión a features pre-extraídas [B, N, D]."""

    def __init__(self, embed_dim, out_dim=1, pool=None):
        super().__init__()
        self.pool = pool or AvgPool()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, out_dim),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = self.pool(x)  # [B, N, D] → [B, D]
        return self.head(x)


# ============================================================================
# Checkpoint loader
# ============================================================================

def load_ijepa2d_encoder(checkpoint_path, model_name='vit_base',
                         patch_size=16, crop_size=224, in_chans=1):
    """Carga el target_encoder de un checkpoint de pretraining I-JEPA 2D.

    El target_encoder se prefiere sobre el context_encoder por su estabilidad
    (EMA weights).  Los pesos del checkpoint pueden tener prefijo 'module.'
    si se entrenaron con DDP.
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Crear encoder fresco
    encoder = vit.__dict__[model_name](
        img_size=[crop_size],
        patch_size=patch_size,
        in_chans=in_chans)
    embed_dim = encoder.embed_dim

    # Cargar target_encoder y eliminar prefijo 'module.' de DDP
    state_dict = ckpt['target_encoder']
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        cleaned[new_k] = v

    msg = encoder.load_state_dict(cleaned, strict=True)
    epoch = ckpt.get('epoch', '?')
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f'Loaded target_encoder from epoch {epoch} ({n_params/1e6:.2f}M params)')
    print(f'  model={model_name}, patch_size={patch_size}, '
          f'crop_size={crop_size}, in_chans={in_chans}, embed_dim={embed_dim}')
    if msg.missing_keys or msg.unexpected_keys:
        print(f'  load msg: {msg}')

    return encoder, embed_dim


# ============================================================================
# Spectra → GADF 2D conversion
# ============================================================================

@torch.no_grad()
def extract_features(encoder, images, device, batch_size=32):
    """Extrae patch tokens: [N, 1, H, W] → [N, 196, D] en CPU."""
    encoder.eval()
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        feats = encoder(batch, masks=None)   # [B, 196, D]
        all_feats.append(feats.cpu())
    return torch.cat(all_feats, dim=0)


def spectra_to_gadf(spectra_array, image_size=224):
    """Convierte un array de espectros [N, L] a imágenes GADF [N, 1, H, W].

    Usa pyts.image.GramianAngularField con método 'difference'.
    """
    gaf = GramianAngularField(image_size=image_size, method='difference')
    gadf_images = gaf.transform(spectra_array)  # [N, H, W]
    return torch.from_numpy(gadf_images.astype(np.float32)).unsqueeze(1)  # [N, 1, H, W]


# ============================================================================
# SimpleTask2D — carga espectros, convierte a GADF 2D
# ============================================================================

class SimpleTask2D:
    """Carga datos de X_supp / y_supp y convierte espectros a GADF 2D.

    Soporta dos modos (auto-detectado):
    - Si existen X_supp.pt / X_query.pt: carga tensores pre-computados
    - Si no: lee X_supp.csv / X_query.csv y computa GADF on-the-fly

    IMPORTANTE: Los espectros X se pasan RAW a GADF (sin Savgol, sin resampling,
    sin escalar), replicando exactamente el pipeline de pretraining
    (precompute_gadf.py).  Solo los targets y se escalan con StandardScaler.
    """

    def __init__(self, data_path, target_column=None,
                 image_size=224, norm_stats=None, scale_y=True,
                 device='cuda', global_stats=None, savgol_params=None):
        self.name = os.path.basename(data_path.rstrip('/'))
        self.data_path = data_path
        self.device = device
        self.target_column = target_column

        # -- Cargar targets
        support_y = pd.read_csv(os.path.join(data_path, "y_supp.csv"), index_col=0)
        query_y = pd.read_csv(os.path.join(data_path, "y_query.csv"), index_col=0)

        # -- Seleccionar columna target
        if target_column is not None:
            support_y = support_y[target_column]
            query_y = query_y[target_column]
        else:
            numeric_cols = support_y.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                support_y = support_y[numeric_cols[0]]
                query_y = query_y[numeric_cols[0]]
            else:
                support_y = support_y.iloc[:, 0]
                query_y = query_y.iloc[:, 0]

        # -- Filtrar NaN
        supp_valid = ~support_y.isna()
        query_valid = ~query_y.isna()
        support_y = support_y[supp_valid].reset_index(drop=True)
        query_y = query_y[query_valid].reset_index(drop=True)

        print(f"  Filtered NaN: support {supp_valid.sum()}/{len(supp_valid)}, "
              f"query {query_valid.sum()}/{len(query_valid)}")

        # -- Cargar X: pre-computado (.pt) o on-the-fly (.csv → GADF)
        supp_pt = os.path.join(data_path, "X_supp.pt")
        query_pt = os.path.join(data_path, "X_query.pt")

        if os.path.exists(supp_pt) and os.path.exists(query_pt):
            # Modo pre-computado: cargar tensores directamente
            self.support_x = torch.load(supp_pt, map_location="cpu").float()[supp_valid.values]
            self.query_x = torch.load(query_pt, map_location="cpu").float()[query_valid.values]
            print(f"  Loaded precomputed GADF: support {tuple(self.support_x.shape)}, "
                  f"query {tuple(self.query_x.shape)}")
        else:
            # Modo on-the-fly: leer CSV de espectros y computar GADF
            support_x = pd.read_csv(os.path.join(data_path, "X_supp.csv"), index_col=0)
            query_x = pd.read_csv(os.path.join(data_path, "X_query.csv"), index_col=0)
            support_x = support_x[supp_valid].reset_index(drop=True)
            query_x = query_x[query_valid].reset_index(drop=True)

            supp_np = support_x.values.astype(np.float32)
            query_np = query_x.values.astype(np.float32)
            if savgol_params is not None:
                from src.gadf_utils import apply_savitzky_golay
                supp_np = apply_savitzky_golay(supp_np, **savgol_params)
                query_np = apply_savitzky_golay(query_np, **savgol_params)
            print(f"  Computing GADF ({image_size}x{image_size}) for support ({supp_np.shape[0]}) "
                  f"and query ({query_np.shape[0]}) on-the-fly ...")
            self.support_x = spectra_to_gadf(supp_np, image_size=image_size)
            self.query_x = spectra_to_gadf(query_np, image_size=image_size)
            if global_stats is not None:
                from src.gadf_utils import encode_diagonal
                gmin, gmax = global_stats
                encode_diagonal(self.support_x[:, 0].numpy(), supp_np,
                                gmin, gmax, image_size)
                encode_diagonal(self.query_x[:, 0].numpy(), query_np,
                                gmin, gmax, image_size)

        # -- Normalizar GADF (misma normalización que el pretraining)
        if norm_stats is not None:
            mean, std = norm_stats
            self.support_x = (self.support_x - mean) / std
            self.query_x = (self.query_x - mean) / std

        # -- Targets: escalar con StandardScaler (fit en support, transform en query)
        supp_y_vals = support_y.values.reshape(-1, 1).astype(np.float32)
        query_y_vals = query_y.values.reshape(-1, 1).astype(np.float32)

        # Pool unificado de y RAW (antes de escalar) para K-fold CV
        self.full_y_raw = np.concatenate([supp_y_vals, query_y_vals], axis=0).ravel()
        self._scale_y = scale_y

        if scale_y:
            self.y_scaler = StandardScaler()
            supp_y_vals = self.y_scaler.fit_transform(supp_y_vals)
            query_y_vals = self.y_scaler.transform(query_y_vals)
        else:
            self.y_scaler = None

        self.support_y = torch.from_numpy(supp_y_vals)
        self.query_y = torch.from_numpy(query_y_vals)

        # Pool unificado de X para K-fold CV (concatenar support y query ya normalizados)
        self.full_x = torch.cat([self.support_x, self.query_x], dim=0)

        print(f"  Loaded SimpleTask2D '{self.name}': "
              f"support={self.support_x.shape}, query={self.query_x.shape}, "
              f"full_pool={self.full_x.shape[0]}")

    def sample_fixed(self, shots, queries):
        """Muestrea soporte/query con splits fijos reproducibles."""
        region_supp_file = os.path.join(self.data_path, f"fixed_val_support_{shots}shots.csv")
        region_query_file = os.path.join(self.data_path, f"fixed_val_query_{shots}shots.csv")

        if os.path.exists(region_supp_file):
            supp_df = pd.read_csv(region_supp_file, index_col=0)
            support_idx = supp_df['support_idx'].values
            support_idx = support_idx[support_idx < self.support_x.shape[0]]

            if os.path.exists(region_query_file):
                query_df = pd.read_csv(region_query_file, index_col=0)
                query_idx = query_df['query_idx'].values
                query_idx = query_idx[query_idx < self.query_x.shape[0]]
            else:
                n_query = min(queries, self.query_x.shape[0])
                query_idx = np.random.choice(self.query_x.shape[0], n_query, replace=False)

            return {
                'support_features': self.support_x[support_idx].to(self.device),
                'support_targets': self.support_y[support_idx].to(self.device),
                'query_features': self.query_x[query_idx].to(self.device),
                'query_targets': self.query_y[query_idx].to(self.device),
            }

        # Fallback: generar split fijo con el mismo formato que fixed_val_support/query_*shots.csv
        supp_fallback = os.path.join(self.data_path, f"fixed_val_support_{shots}shots.csv")
        query_fallback = os.path.join(self.data_path, f"fixed_val_query_{shots}shots.csv")

        regenerate = False
        if os.path.exists(supp_fallback) and os.path.exists(query_fallback):
            support_idx = pd.read_csv(supp_fallback, index_col=0)['support_idx'].values
            query_idx = pd.read_csv(query_fallback, index_col=0)['query_idx'].values
            if support_idx.max() >= self.support_x.shape[0] or query_idx.max() >= self.query_x.shape[0]:
                regenerate = True
        else:
            regenerate = True

        if regenerate:
            n_support = min(shots, self.support_x.shape[0])
            n_query = min(queries, self.query_x.shape[0])
            support_idx = np.random.choice(self.support_x.shape[0], n_support, replace=False)
            query_idx = np.random.choice(self.query_x.shape[0], n_query, replace=False)
            pd.DataFrame({'support_idx': support_idx}).to_csv(supp_fallback)
            pd.DataFrame({'query_idx': query_idx}).to_csv(query_fallback)

        return {
            'support_features': self.support_x[support_idx].to(self.device),
            'support_targets': self.support_y[support_idx].to(self.device),
            'query_features': self.query_x[query_idx].to(self.device),
            'query_targets': self.query_y[query_idx].to(self.device),
        }

    def iter_folds(self, k_spt, seed=42):
        """K-fold CV sobre el pool unificado support∪query.

        Yields K = ceil(N / k_spt) folds; cada instancia es support exactamente
        una vez y query en todos los demás folds.  El StandardScaler se re-ajusta
        por fold sólo sobre el support de ese fold.
        """
        N = self.full_x.shape[0]
        if N < k_spt:
            print(f"  [CV] WARN: task '{self.name}' sólo tiene {N} instancias "
                  f"({k_spt} requeridas para support). Saltando.")
            return

        rng = np.random.RandomState(seed)
        perm = rng.permutation(N)
        K = (N + k_spt - 1) // k_spt  # ceil

        for fold_idx in range(K):
            spt_idx = perm[fold_idx * k_spt: (fold_idx + 1) * k_spt]
            qry_idx = np.setdiff1d(perm, spt_idx, assume_unique=True)

            spt_y_raw = self.full_y_raw[spt_idx].reshape(-1, 1).astype(np.float32)
            qry_y_raw = self.full_y_raw[qry_idx].reshape(-1, 1).astype(np.float32)

            if self._scale_y:
                fold_scaler = StandardScaler()
                spt_y = fold_scaler.fit_transform(spt_y_raw).astype(np.float32)
                qry_y = fold_scaler.transform(qry_y_raw).astype(np.float32)
            else:
                fold_scaler = None
                spt_y = spt_y_raw
                qry_y = qry_y_raw

            yield {
                'fold_idx': fold_idx,
                'num_folds': K,
                'support_features': self.full_x[spt_idx].to(self.device),
                'support_targets': torch.from_numpy(spt_y).to(self.device),
                'query_features': self.full_x[qry_idx].to(self.device),
                'query_targets': torch.from_numpy(qry_y).to(self.device),
                'y_scaler': fold_scaler,
            }

    def query_dataloader(self, batch_size=32):
        indices = torch.arange(self.query_x.shape[0])
        dataset = TensorDataset(self.query_x, self.query_y, indices)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ============================================================================
# Auto-detect target properties
# ============================================================================

def get_dataset_properties(data_path):
    y_path = os.path.join(data_path, 'y_supp.csv')
    y = pd.read_csv(y_path, index_col=0)
    exclude_cols = {'dataset'}
    valid_props = []
    for col in y.columns:
        col_clean = col.strip()
        if col_clean in exclude_cols:
            continue
        if pd.to_numeric(y[col], errors='coerce').notna().sum() > len(y) * 0.05:
            valid_props.append(col_clean)
    return valid_props


# ============================================================================
# Training helpers
# ============================================================================

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device,
                    val_loss, scheduler_type):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_mae += torch.abs(pred - y).sum().item()
        n += x.size(0)

    if scheduler_type == 'ReduceLROnPlateau':
        scheduler.step(val_loss)
    else:
        scheduler.step()

    return total_loss / n, total_mae / n


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        pred = model(x)
        total_loss += loss_fn(pred, y).item() * x.size(0)
        total_mae += torch.abs(pred - y).sum().item()
        n += x.size(0)
    rmse = np.sqrt(total_loss / n)
    return total_loss / n, total_mae / n, rmse


# ============================================================================
# Argument parser
# ============================================================================

def get_args_parser():
    p = argparse.ArgumentParser('GADF-2D I-JEPA Fine-tuning', add_help=False)

    # Data
    p.add_argument('--data_path', default='../SpectraI-JEPA/data/Soil_NIR_AGG_mixed', type=str)
    p.add_argument('--split', default='test', type=str)
    p.add_argument('--device', default='cuda', type=str)

    # Training
    p.add_argument('--epoch', default=1000, type=int)
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--lr', default=1e-4, type=float)
    p.add_argument('--wd', default=1e-5, type=float)
    p.add_argument('--scheduler_type', default='ReduceLROnPlateau', type=str)
    p.add_argument('--patience', default=30, type=int)
    p.add_argument('--linear_probing', action='store_true',
                   help='Freeze encoder, train only regression head')
    p.add_argument('--pool_mode', default='avg', choices=['avg', 'attn', 'map'],
                   help='Pooling over patch tokens: avg (mean), attn (learned weights), map (cross-attention)')
    p.add_argument('--use_lora', action='store_true',
                   help='Use LoRA fine-tuning (freeze base, train LoRA + head)')
    p.add_argument('--lora_r', type=int, default=8,
                   help='LoRA rank (default: 8)')
    p.add_argument('--lora_alpha', type=float, default=16.0,
                   help='LoRA alpha scaling (default: 16.0)')
    p.add_argument('--lora_dropout', type=float, default=0.05,
                   help='LoRA dropout (default: 0.05)')
    p.add_argument('--lora_target_modules', type=str, nargs='+',
                   default=['attn.qkv', 'attn.proj'],
                   help='Module name substrings to apply LoRA to')

    # Output
    p.add_argument('--save_path', default='results/ft_gadf2d/', type=str)

    # Pretrained checkpoint
    p.add_argument('--model_weight', default='', type=str,
                   help='Path to I-JEPA 2D pretraining checkpoint (.pth.tar)')

    # Model architecture (fallback cuando no hay checkpoint)
    p.add_argument('--model_name', default='vit_base', type=str)
    p.add_argument('--patch_size', default=16, type=int)
    p.add_argument('--crop_size', default=224, type=int)

    # GADF
    p.add_argument('--gadf_image_size', default=224, type=int,
                   help='Tamaño de la imagen GADF generada')
    p.add_argument('--gadf_norm_stats', default='data/gadf_norm_stats.json',
                   type=str, help='JSON con mean/std de normalización GADF '
                   '(generado por compute_gadf_stats.py)')
    p.add_argument('--encode_diagonal', action='store_true',
                   help='Codifica magnitud espectral en la diagonal GADF')
    p.add_argument('--gadf_global_stats', default='data/gadf_paa_global_stats.json',
                   type=str, help='Ruta al JSON con min/max globales PAA')
    p.add_argument('--savgol', action='store_true',
                   help='Aplica filtro Savitzky-Golay antes de la transformación GADF (solo modo on-the-fly)')
    p.add_argument('--savgol_window', type=int, default=15,
                   help='Longitud de ventana SG (impar, default: 15)')
    p.add_argument('--savgol_polyorder', type=int, default=2,
                   help='Orden del polinomio SG (default: 2)')
    p.add_argument('--savgol_deriv', type=int, default=0,
                   help='Derivada SG: 0=suavizado, 1=primera derivada (default: 0)')

    # Few-shot
    p.add_argument('--k_spt', type=int, default=25)
    p.add_argument('--k_qry', type=int, default=25)
    p.add_argument('--no_scale_y', action='store_true',
                   help='Disable target StandardScaler')

    # Dataset modes
    p.add_argument('--simple_dataset', action='store_true')
    p.add_argument('--target_column', default=None, type=str)
    p.add_argument('--region_tasks', action='store_true')

    # Repeats / eval mode
    p.add_argument('--n_repeats', type=int, default=3)
    p.add_argument('--eval_mode', type=str, default='fixed',
                   choices=['fixed', 'cv'],
                   help='fixed: usa fixed_val_*shots.csv (legacy); '
                        'cv: K-fold CV sobre pool unificado support∪query')

    return p


# ============================================================================
# Main
# ============================================================================

def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    if args.use_lora and args.linear_probing:
        print("ERROR: --use_lora and --linear_probing are mutually exclusive")
        sys.exit(1)

    device = torch.device(args.device)
    print(f'training device: {device}')

    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # -- Normalización GADF (lee de JSON, selecciona variante según --encode_diagonal)
    if os.path.exists(args.gadf_norm_stats):
        from src.gadf_utils import load_norm_stats
        norm_mean, norm_std = load_norm_stats(args.gadf_norm_stats,
                                              use_diagonal=args.encode_diagonal)
        norm_stats = (norm_mean[0], norm_std[0])  # scalars for SimpleTask2D
        variant = 'with_diagonal' if args.encode_diagonal else 'no_diagonal'
        print(f"Norm stats from {args.gadf_norm_stats} [{variant}]: "
              f"mean={norm_stats[0]:.6f}, std={norm_stats[1]:.6f}")
    else:
        norm_stats = (-0.0000, 0.5922)
        print(f"WARNING: {args.gadf_norm_stats} not found, using hardcoded defaults")
    scale_y = not args.no_scale_y

    # -- Diagonal encoding (opcional)
    global_stats = None
    if args.encode_diagonal:
        from src.gadf_utils import load_global_stats
        global_stats = load_global_stats(args.gadf_global_stats)
        print(f"Diagonal encoding ON (min={global_stats[0]:.4f}, max={global_stats[1]:.4f})")

    # -- Savitzky-Golay (solo path on-the-fly; ignorado si existen .pt precomputados)
    savgol_params = None
    if args.savgol:
        savgol_params = dict(window_length=args.savgol_window,
                             polyorder=args.savgol_polyorder,
                             deriv=args.savgol_deriv)
        print(f"Savitzky-Golay ON (window={args.savgol_window}, poly={args.savgol_polyorder}, deriv={args.savgol_deriv})")

    # -- Cargar tareas
    if args.region_tasks:
        splits_file = os.path.join(args.data_path, 'splits.csv')
        if os.path.exists(splits_file):
            splits_df = pd.read_csv(splits_file, index_col=0)
            task_names = splits_df[splits_df['split'] == args.split]['task'].tolist()
            print(f"Using splits.csv: {len(task_names)} tasks for split '{args.split}'")
        else:
            task_names = sorted([d for d in os.listdir(args.data_path)
                                 if os.path.isdir(os.path.join(args.data_path, d))])
            print(f"No splits.csv found, using all {len(task_names)} subdirectories")

        tasks = []
        for tname in task_names:
            task_dir = os.path.join(args.data_path, tname)
            if not os.path.isdir(task_dir):
                continue
            print(f"\nLoading region task: {tname}")
            try:
                t = SimpleTask2D(
                    data_path=task_dir, target_column=None,
                    image_size=args.gadf_image_size,
                    norm_stats=norm_stats, scale_y=scale_y,
                    device=device, global_stats=global_stats,
                    savgol_params=savgol_params)
                t.name = tname
                tasks.append(t)
            except Exception as e:
                print(f"  Skipping {tname}: {e}")
        print(f"\nLoaded {len(tasks)} region task(s)")

    elif args.simple_dataset:
        if args.target_column is not None:
            target_columns = [args.target_column]
        else:
            target_columns = get_dataset_properties(args.data_path)
            print(f"Auto-detected {len(target_columns)} properties: {target_columns}")

        tasks = []
        for tc in target_columns:
            print(f"\nLoading simple dataset from: {args.data_path} (target: {tc})")
            t = SimpleTask2D(
                data_path=args.data_path, target_column=tc,
                image_size=args.gadf_image_size,
                norm_stats=norm_stats, scale_y=scale_y,
                device=device, global_stats=global_stats,
                savgol_params=savgol_params)
            t.name = f"{t.name}_{tc}"
            tasks.append(t)
        print(f"\nLoaded {len(tasks)} task(s) in simple mode")

    else:
        print("ERROR: Especifica --region_tasks o --simple_dataset")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Task loop
    # -------------------------------------------------------------------
    results_per_task = []
    predictions_all_tasks = []
    n_repeats = args.n_repeats
    eval_mode = args.eval_mode

    if eval_mode == 'cv':
        print(f"\nEval mode: K-fold CV (k_spt={args.k_spt}, seed=42)")
        if n_repeats != 3:
            print("  WARNING: --n_repeats ignorado en modo cv (las repeticiones son los folds)")
    else:
        print(f"\nEval mode: fixed — {n_repeats} repeat(s) per task")

    for task_idx, task in enumerate(tasks):
        task_name = task.name
        print(f"\n{'='*60}")
        print(f"Task {task_idx + 1}/{len(tasks)}: {task_name}")
        print(f"{'='*60}")

        # -- Linear probing: pre-extraer features UNA vez por tarea
        if args.linear_probing:
            if args.model_weight:
                encoder, embed_dim = load_ijepa2d_encoder(
                    checkpoint_path=args.model_weight,
                    model_name=args.model_name,
                    patch_size=args.patch_size,
                    crop_size=args.crop_size,
                    in_chans=1)
            else:
                encoder = vit.__dict__[args.model_name](
                    img_size=[args.crop_size],
                    patch_size=args.patch_size,
                    in_chans=1)
                embed_dim = encoder.embed_dim
                print('Using random encoder (no pretrained weights)')

            encoder = encoder.to(device)
            print(f"  Pre-extracting features for linear probing...")
            task.support_x = extract_features(
                encoder, task.support_x, device, batch_size=args.batch_size)
            task.query_x = extract_features(
                encoder, task.query_x, device, batch_size=args.batch_size)
            print(f"  Features: support {tuple(task.support_x.shape)}, "
                  f"query {tuple(task.query_x.shape)}")

            # En modo cv, full_x también debe contener features extraídas
            if eval_mode == 'cv':
                task.full_x = torch.cat([task.support_x, task.query_x], dim=0)

            del encoder
            torch.cuda.empty_cache()

        # Generar iterador de episodios según eval_mode
        if eval_mode == 'cv':
            def _episodes():
                for fold in task.iter_folds(args.k_spt, seed=42):
                    yield fold['fold_idx'] + 1, fold['num_folds'], fold
            episode_iter = _episodes()
        else:
            def _episodes():
                for repeat in range(1, n_repeats + 1):
                    torch.manual_seed(42 + repeat)
                    np.random.seed(42 + repeat)
                    data = task.sample_fixed(args.k_spt, args.k_qry)
                    yield repeat, n_repeats, data
            episode_iter = _episodes()

        for repeat, total_episodes, data in episode_iter:
            if eval_mode == 'cv':
                print(f"\n--- Fold {repeat}/{total_episodes} ---")
            else:
                print(f"\n--- Repeat {repeat}/{total_episodes} ---")

            support_x = data['support_features']
            support_y = data['support_targets']
            query_x = data['query_features']
            query_y = data['query_targets']
            print(f"Support: {support_x.shape[0]}, Query: {query_x.shape[0]}")

            support_loader = DataLoader(
                TensorDataset(support_x, support_y),
                batch_size=args.batch_size, shuffle=True)
            query_loader = DataLoader(
                TensorDataset(query_x, query_y),
                batch_size=args.batch_size, shuffle=False)

            # -- Build model
            if args.linear_probing:
                # Features ya pre-extraídas → pool + cabeza
                pool = make_pool(args.pool_mode, embed_dim)
                model = HeadOnlyModel(embed_dim, out_dim=1, pool=pool).to(device)
                trainable = sum(p.numel() for p in model.parameters())
                print(f'Linear probing (pre-extracted, pool={args.pool_mode}): '
                      f'{trainable} trainable params')
            else:
                if args.model_weight:
                    encoder, embed_dim = load_ijepa2d_encoder(
                        checkpoint_path=args.model_weight,
                        model_name=args.model_name,
                        patch_size=args.patch_size,
                        crop_size=args.crop_size,
                        in_chans=1)
                else:
                    encoder = vit.__dict__[args.model_name](
                        img_size=[args.crop_size],
                        patch_size=args.patch_size,
                        in_chans=1)
                    embed_dim = encoder.embed_dim
                    print('Training from scratch (no pretrained weights)')

                pool = make_pool(args.pool_mode, embed_dim)
                model = GADF2DForRegression(encoder, embed_dim, out_dim=1, pool=pool).to(device)
                total_params = sum(p.numel() for p in model.parameters())
                print(f'GADF2DForRegression: {total_params/1e6:.2f}M params')

                if args.use_lora:
                    from src.lora import apply_lora, count_trainable_parameters
                    for param in model.encoder.parameters():
                        param.requires_grad_(False)
                    n_lora = apply_lora(model.encoder,
                                        target_modules=args.lora_target_modules,
                                        r=args.lora_r, alpha=args.lora_alpha,
                                        dropout=args.lora_dropout)
                    model.to(device)  # move new LoRA params to device
                    trainable, total, pct = count_trainable_parameters(model)
                    print(f'LoRA applied to {n_lora} modules: '
                          f'{trainable} trainable / {total} total ({pct:.2f}%)')

            # -- Optimizer
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, betas=(0.9, 0.95), weight_decay=args.wd)

            if args.scheduler_type == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3,
                                              verbose=False, threshold=0.0001,
                                              patience=10)
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

            loss_fn = nn.MSELoss().to(device)
            es = EarlyStop(patience=args.patience, mode='min')
            val_loss = float('inf')
            loss_history = []

            # -- Training
            for epoch in range(1, args.epoch + 1):
                train_loss, train_mae = train_one_epoch(
                    model, support_loader, optimizer, scheduler,
                    loss_fn, device, val_loss, args.scheduler_type)
                val_loss, val_mae, val_rmse = evaluate(
                    model, query_loader, loss_fn, device)
                loss_history.append({
                    'epoch': epoch, 'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_mae': train_mae, 'val_mae': val_mae,
                })

                if epoch % 50 == 0 or epoch == 1:
                    print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, "
                          f"val_loss={val_loss:.6f}, val_rmse={val_rmse:.6f}")

                es(val_loss, model,
                   os.path.join(args.save_path, f"{task_name}_best.pth"),
                   epoch)
                if es.early_stop:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(best epoch {es.best_epoch})")
                    break

            # -- Loss curves (first repeat only)
            if repeat == 1:
                curves_dir = os.path.join(args.save_path, 'loss_curves')
                os.makedirs(curves_dir, exist_ok=True)
                loss_df = pd.DataFrame(loss_history)
                loss_df.to_csv(os.path.join(curves_dir, f'{task_name}_loss.csv'),
                               index=False)

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.plot(loss_df['epoch'], loss_df['train_loss'],
                        label='Train Loss', linewidth=1.5)
                ax.plot(loss_df['epoch'], loss_df['val_loss'],
                        label='Val Loss', linewidth=1.5)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss (MSE)')
                ax.set_title(f'Loss Curve — {task_name}')
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(curves_dir, f'{task_name}_loss.png'),
                            dpi=150)
                plt.close(fig)

            # -- Evaluate best model on query set
            best_path = os.path.join(args.save_path, f"{task_name}_best.pth")
            try:
                model.load_state_dict(torch.load(best_path, map_location=device))
            except RuntimeError as e:
                print(f"WARNING: Could not load best checkpoint ({e}). "
                      f"Using last model state instead.")
            model.eval()

            y_true, y_pred = [], []
            mse_total, mae_total, num_instances = 0.0, 0.0, 0

            if eval_mode == 'cv':
                # Evaluar sobre el query set del fold (ya está en tensores)
                eval_dl = DataLoader(
                    TensorDataset(query_x, query_y),
                    batch_size=args.batch_size, shuffle=False)
                with torch.no_grad():
                    for x, y in eval_dl:
                        x, y = x.to(device), y.to(device)
                        outputs = model(x)
                        mse_total += ((outputs - y) ** 2).sum().item()
                        mae_total += torch.abs(outputs - y).sum().item()
                        num_instances += y.size(0)
                        y_true.extend(y.cpu().numpy().flatten().tolist())
                        y_pred.extend(outputs.cpu().numpy().flatten().tolist())
            else:
                # Modo fixed: evaluar sobre el query set completo original
                query_dl = task.query_dataloader()
                with torch.no_grad():
                    for x, y, idx in query_dl:
                        x, y = x.to(device), y.to(device)
                        outputs = model(x)
                        mse_total += ((outputs - y) ** 2).sum().item()
                        mae_total += torch.abs(outputs - y).sum().item()
                        num_instances += y.size(0)
                        y_true.extend(y.cpu().numpy().flatten().tolist())
                        y_pred.extend(outputs.cpu().numpy().flatten().tolist())

            mse = mse_total / num_instances
            mae = mae_total / num_instances
            rmse = np.sqrt(mse)

            from sklearn.metrics import r2_score
            r2 = r2_score(y_true, y_pred)

            label = f"Fold {repeat}/{total_episodes}" if eval_mode == 'cv' else f"Repeat {repeat}"
            print(f"\n  {label} Results for {task_name}:")
            print(f"  MSE: {mse:.6f} | MAE: {mae:.6f} | "
                  f"RMSE: {rmse:.6f} | R2: {r2:.6f}")

            results_per_task.append({
                'task': task_name, 'repeat': repeat,
                'mse': mse, 'mae': mae, 'rmse': rmse, 'r2': r2,
                'support_size': support_x.shape[0],
                'query_size': query_x.shape[0],
            })
            predictions_all_tasks.append(pd.DataFrame({
                'task': task_name, 'repeat': repeat,
                'y_true': y_true, 'y_pred': y_pred,
            }))

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    results_df = pd.DataFrame(results_per_task)
    if results_df.empty:
        print("WARNING: No task results — all tasks were skipped. Check data_path and splits.")
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

    final_df = pd.concat([results_df, pd.DataFrame(summary_rows)],
                         ignore_index=True)
    final_df.to_csv(os.path.join(args.save_path, 'results_per_task.csv'),
                    index=False)

    # Resumen por repetición/fold
    if eval_mode == 'cv':
        # En cv los folds varían por tarea; resumen por fold sobre todas las tareas
        fold_ids = sorted(results_df['repeat'].unique())
        repeat_summary_rows = []
        for rep in fold_ids:
            rep_data = results_df[results_df['repeat'] == rep]
            row = {'fold': rep, 'n_tasks': len(rep_data)}
            for col in metric_cols:
                row[col] = rep_data[col].mean()
            repeat_summary_rows.append(row)
        repeat_summary_df = pd.DataFrame(repeat_summary_rows)
        mean_row = {'fold': 'MEAN', 'n_tasks': repeat_summary_df['n_tasks'].mean()}
        for col in metric_cols:
            mean_row[col] = repeat_summary_df[col].mean()
        repeat_summary_df = pd.concat([repeat_summary_df,
                                        pd.DataFrame([mean_row])],
                                       ignore_index=True)
        repeat_summary_df.to_csv(os.path.join(args.save_path,
                                               'results_per_fold.csv'),
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
        mean_row = {'repeat': 'MEAN'}
        for col in metric_cols:
            mean_row[col] = repeat_summary_df[col].mean()
        repeat_summary_df = pd.concat([repeat_summary_df,
                                        pd.DataFrame([mean_row])],
                                       ignore_index=True)
        repeat_summary_df.to_csv(os.path.join(args.save_path,
                                               'results_per_repeat.csv'),
                                 index=False)

    if predictions_all_tasks:
        predictions_df = pd.concat(predictions_all_tasks, ignore_index=True)
        predictions_df.to_csv(os.path.join(args.save_path,
                                            'predictions_all_tasks.csv'),
                              index=False)

    print(f"\n{'='*60}")
    print("OVERALL RESULTS:")
    print(f"{'='*60}")
    print(f"Average MSE:  {avg_mean['mse']:.6f} +/- {avg_std['mse']:.6f}")
    print(f"Average MAE:  {avg_mean['mae']:.6f} +/- {avg_std['mae']:.6f}")
    print(f"Average RMSE: {avg_mean['rmse']:.6f} +/- {avg_std['rmse']:.6f}")
    print(f"Average R2:   {avg_mean['r2']:.6f} +/- {avg_std['r2']:.6f}")
    print(f"\nResults saved to {args.save_path}")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
