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

    Loads emb_supp.pt [N_s, D] and emb_query.pt [N_q, D] plus target CSVs.
    Mirrors the sample() / sample_fixed() / query_dataloader() API of MixedTask.
    """

    def __init__(self, task_dir, name, device='cuda', scale_y=True):
        self.name = name
        self.task_dir = task_dir
        self.device = device

        # Load embeddings
        self.support_x = torch.load(
            os.path.join(task_dir, 'emb_supp.pt'), map_location='cpu').float()
        self.query_x = torch.load(
            os.path.join(task_dir, 'emb_query.pt'), map_location='cpu').float()

        # Load targets — pick first numeric column
        sy = pd.read_csv(os.path.join(task_dir, 'y_supp.csv'), index_col=0)
        qy = pd.read_csv(os.path.join(task_dir, 'y_query.csv'), index_col=0)
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
            self.task_dir, f'fixed_val_support_{shots}shots.csv')
        query_csv = os.path.join(
            self.task_dir, f'fixed_val_query_{shots}shots.csv')

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

    def query_dataloader(self, batch_size=32):
        """Full query set DataLoader for evaluation."""
        ds = _QueryDataset(self.query_x, self.query_y, self.query_idx)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)


class EmbeddingDataset:
    """Collection of EmbeddingTasks filtered by train/val/test split.

    Mirrors MixedDataset from fewshot_nir_orig/datasets_fewshot.py.
    """

    def __init__(self, path, split='train', device='cuda',
                 scale_y=True, max_tasks=None):
        self.path = path
        self.split = split
        self.device = device

        # Read splits
        splits_df = pd.read_csv(os.path.join(path, 'splits.csv'), index_col=0)
        task_names = splits_df.query(
            f"split == '{split}'")['task'].values.tolist()

        if max_tasks is not None and len(task_names) > max_tasks:
            task_names = list(np.random.choice(
                task_names, max_tasks, replace=False))

        # Create tasks
        self.tasks = {}
        self.task_names = []
        for name in sorted(task_names):
            task_dir = os.path.join(path, name)
            emb_supp = os.path.join(task_dir, 'emb_supp.pt')
            emb_query = os.path.join(task_dir, 'emb_query.pt')
            if not os.path.exists(emb_supp) or not os.path.exists(emb_query):
                continue
            self.tasks[name] = EmbeddingTask(
                task_dir, name, device=device, scale_y=scale_y)
            self.task_names.append(name)

        print(f'EmbeddingDataset [{split}]: {len(self.tasks)} tasks loaded')

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        name = self.task_names[idx]
        return self.tasks[name]

    def __iter__(self):
        for name in self.task_names:
            yield self.tasks[name]
