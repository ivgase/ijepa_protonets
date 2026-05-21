# ijepa_spectra2d — Dataset classes for precomputed I-JEPA embeddings
#
# EmbeddingTask / EmbeddingDataset replicate the API of MixedTask / MixedDataset
# from fewshot_nir_orig/datasets_fewshot.py, but load emb_supp.pt / emb_query.pt
# (flat [N, D] tensors) instead of raw spectra.

import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class _QueryDataset(Dataset):
    """Simple wrapper for full query set evaluation."""

    def __init__(self, x, y, idx):
        self.x = x
        self.y = y
        self.idx = idx

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.idx[i]


class EmbeddingTask:
    """A single task with precomputed I-JEPA embeddings.

    Loads embeddings plus target CSVs.  Two modes via emb_mode:
      'mean':    emb_supp.pt    [N_s, D]       — mean-pooled patch tokens
      'patches': emb_supp_patches.pt [N_s, P, D] — all patch tokens kept

    Mirrors the sample() / sample_fixed() / query_dataloader() API of MixedTask.
    """

    def __init__(self, data_task_dir, name, embedding_task_dir=None,
                 device='cuda', scale_y=True, emb_mode='mean'):
        self.name = name
        self.data_task_dir = data_task_dir
        self.embedding_task_dir = embedding_task_dir or data_task_dir
        self.device = device

        # Load embeddings (filename depends on mode)
        suffix = '_patches' if emb_mode == 'patches' else ''
        self.support_x = torch.load(
            os.path.join(self.embedding_task_dir, f'emb_supp{suffix}.pt'),
            map_location='cpu').float()
        self.query_x = torch.load(
            os.path.join(self.embedding_task_dir, f'emb_query{suffix}.pt'),
            map_location='cpu').float()

        # Load targets — pick first numeric column
        sy = pd.read_csv(os.path.join(self.data_task_dir, 'y_supp.csv'), index_col=0)
        qy = pd.read_csv(os.path.join(self.data_task_dir, 'y_query.csv'), index_col=0)
        num_cols = sy.select_dtypes(include='number').columns
        col = num_cols[0]
        sy = sy[col]
        qy = qy[col]

        # Drop NaN
        supp_valid = ~sy.isna()
        query_valid = ~qy.isna()
        self.support_x = self.support_x[supp_valid.values]
        self.query_x = self.query_x[query_valid.values]
        sy = sy[supp_valid].values.astype(np.float32)
        qy = qy[query_valid].values.astype(np.float32)

        # Scale targets (fit on support, transform on query)
        if scale_y:
            self.scaler = StandardScaler()
            sy = self.scaler.fit_transform(sy.reshape(-1, 1)).ravel()
            qy = self.scaler.transform(qy.reshape(-1, 1)).ravel()

        self.support_y = torch.from_numpy(sy).unsqueeze(1)  # [N_s, 1]
        self.query_y = torch.from_numpy(qy).unsqueeze(1)    # [N_q, 1]

        self.support_idx = torch.arange(self.support_x.shape[0])
        self.query_idx = torch.arange(self.query_x.shape[0])

        # Pool unificado para K-fold CV
        self._scale_y = scale_y
        # Recuperar y raw (antes de escalar): desescalar support y query si aplicó scaler
        if scale_y:
            sy_raw = self.scaler.inverse_transform(sy.reshape(-1, 1)).ravel()
            qy_raw = self.scaler.inverse_transform(qy.reshape(-1, 1)).ravel()
        else:
            sy_raw = sy
            qy_raw = qy
        self.full_y_raw = np.concatenate([sy_raw, qy_raw], axis=0).astype(np.float32)
        self.full_x = torch.cat([self.support_x, self.query_x], dim=0)

    def sample(self, shots, queries):
        """Random sample for episodic training."""
        n_s = self.support_x.shape[0]
        n_q = self.query_x.shape[0]
        if n_s < shots or n_q < queries:
            s = min(shots, n_s)
            q = min(queries, n_q)
        else:
            s, q = shots, queries

        si = np.random.choice(n_s, s, replace=False)
        qi = np.random.choice(n_q, q, replace=False)

        return {
            'support_features': self.support_x[si].to(self.device),
            'support_targets': self.support_y[si].to(self.device),
            'query_features': self.query_x[qi].to(self.device),
            'query_targets': self.query_y[qi].to(self.device),
        }

    def sample_fixed(self, shots, queries):
        """Deterministic sample for val/test (persists to CSV)."""
        supp_csv = os.path.join(
            self.data_task_dir, f'fixed_val_support_{shots}shots.csv')
        query_csv = os.path.join(
            self.data_task_dir, f'fixed_val_query_{shots}shots.csv')

        if os.path.exists(supp_csv) and os.path.exists(query_csv):
            si = pd.read_csv(supp_csv, index_col=0)['support_idx'].values
            qi = pd.read_csv(query_csv, index_col=0)['query_idx'].values
            # Clip indices to current size (safety)
            si = si[si < self.support_x.shape[0]]
            qi = qi[qi < self.query_x.shape[0]]
        else:
            n_s = self.support_x.shape[0]
            n_q = self.query_x.shape[0]
            s = min(shots, n_s)
            q = min(queries, n_q)
            si = np.random.choice(n_s, s, replace=False)
            qi = np.random.choice(n_q, q, replace=False)
            pd.DataFrame({'support_idx': si}).to_csv(supp_csv)
            pd.DataFrame({'query_idx': qi}).to_csv(query_csv)

        return {
            'support_features': self.support_x[si].to(self.device),
            'support_targets': self.support_y[si].to(self.device),
            'query_features': self.query_x[qi].to(self.device),
            'query_targets': self.query_y[qi].to(self.device),
        }

    def iter_folds(self, k_spt, seed=42):
        """K-fold CV sobre el pool unificado support∪query.

        Yields K = ceil(N / k_spt) folds; cada instancia es support exactamente
        una vez y query en todos los demás folds.  El StandardScaler se re-ajusta
        por fold sólo sobre el support de ese fold.
        """
        N = self.full_x.shape[0]
        if N < k_spt:
            import warnings
            warnings.warn(f"Task '{self.name}' tiene sólo {N} instancias "
                          f"({k_spt} requeridas). Saltando.")
            return

        rng = np.random.RandomState(seed)
        perm = rng.permutation(N)
        K = (N + k_spt - 1) // k_spt  # ceil

        for fold_idx in range(K):
            spt_idx = perm[fold_idx * k_spt: (fold_idx + 1) * k_spt]
            qry_idx = np.setdiff1d(perm, spt_idx, assume_unique=True)

            spt_y_raw = self.full_y_raw[spt_idx].reshape(-1, 1)
            qry_y_raw = self.full_y_raw[qry_idx].reshape(-1, 1)

            if self._scale_y:
                fold_scaler = StandardScaler()
                spt_y = fold_scaler.fit_transform(spt_y_raw).ravel().astype(np.float32)
                qry_y = fold_scaler.transform(qry_y_raw).ravel().astype(np.float32)
            else:
                fold_scaler = None
                spt_y = spt_y_raw.ravel().astype(np.float32)
                qry_y = qry_y_raw.ravel().astype(np.float32)

            yield {
                'fold_idx': fold_idx,
                'num_folds': K,
                'support_features': self.full_x[spt_idx].to(self.device),
                'support_targets': torch.from_numpy(spt_y).unsqueeze(1).to(self.device),
                'query_features': self.full_x[qry_idx].to(self.device),
                'query_targets': torch.from_numpy(qry_y).unsqueeze(1).to(self.device),
                'y_scaler': fold_scaler,
            }

    def query_dataloader(self, batch_size=32):
        """Full query set DataLoader for evaluation."""
        ds = _QueryDataset(self.query_x, self.query_y, self.query_idx)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)


class EmbeddingDataset:
    """Collection of EmbeddingTasks filtered by train/val/test split.

    Mirrors MixedDataset from fewshot_nir_orig/datasets_fewshot.py.
    """

    def __init__(self, path, split='train', device='cuda',
                 scale_y=True, max_tasks=None, emb_mode='mean',
                 embeddings_root=None):
        self.path = path
        self.split = split
        self.device = device
        self.embeddings_root = embeddings_root

        # Read splits
        splits_df = pd.read_csv(os.path.join(path, 'splits.csv'), index_col=0)
        task_names = splits_df.query(
            f"split == '{split}'")['task'].values.tolist()

        if max_tasks is not None and len(task_names) > max_tasks:
            task_names = list(np.random.choice(
                task_names, max_tasks, replace=False))

        # Determine embedding filenames
        suffix = '_patches' if emb_mode == 'patches' else ''

        # Store task dirs (lazy loading — tasks are created on demand)
        self._task_dirs = {}
        self._embedding_task_dirs = {}
        self.task_names = []
        for name in sorted(task_names):
            task_dir = os.path.join(path, name)
            embedding_task_dir = task_dir
            if embeddings_root is not None:
                embedding_task_dir = os.path.join(embeddings_root, name)

            emb_supp = os.path.join(embedding_task_dir, f'emb_supp{suffix}.pt')
            emb_query = os.path.join(embedding_task_dir, f'emb_query{suffix}.pt')
            if not os.path.exists(emb_supp) or not os.path.exists(emb_query):
                continue
            self._task_dirs[name] = task_dir
            self._embedding_task_dirs[name] = embedding_task_dir
            self.task_names.append(name)

        self._device = device
        self._scale_y = scale_y
        self._emb_mode = emb_mode

        print(f'EmbeddingDataset [{split}]: {len(self.task_names)} tasks loaded')

    def _load_task(self, name):
        return EmbeddingTask(
            self._task_dirs[name], name,
            embedding_task_dir=self._embedding_task_dirs[name],
            device=self._device,
            scale_y=self._scale_y,
            emb_mode=self._emb_mode,
        )

    def __len__(self):
        return len(self.task_names)

    def __getitem__(self, idx):
        return self._load_task(self.task_names[idx])

    def __iter__(self):
        for name in self.task_names:
            yield self._load_task(name)
