# ijepa_spectra2d — Regression ProtoNet for precomputed I-JEPA embeddings
#
# Simplified from fewshot_nir_orig/models/protonet.py:
#   - Standard nn.Module (no vars/initialization pattern)
#   - Vectorized distance computation (no per-sample loop)
#   - Operates on flat [B, D] embeddings, not [B, 1, L] spectra

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score


class ProjectionNet(nn.Module):
    """Learnable MLP that maps frozen I-JEPA embeddings to a metric space.

    Architecture: Linear -> LayerNorm -> ReLU -> Linear
    Uses LayerNorm instead of BatchNorm (k=25 samples is too few for BN).
    """

    def __init__(self, input_dim=768, hidden_dim=256, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class IdentityNet(nn.Module):
    """No-op projection for ablation (raw embeddings)."""

    def forward(self, x):
        return x


class EmbeddingProtoNet(nn.Module):
    """Regression ProtoNet operating on precomputed embeddings.

    Prediction: softmax(-cdist(query, support) / temperature) @ y_support
    Loss: MSE
    """

    def __init__(self, projection, device, dist_temperature=0.5, lr=0.001):
        super().__init__()
        self.projection = projection.to(device)
        self.dev = device
        self.dist_temperature = dist_temperature
        params = list(self.projection.parameters())
        if params:
            self.optimizer = torch.optim.Adam(params, lr=lr)
        else:
            self.optimizer = None  # IdentityNet (no projection)
        self.task_counter = 0

    def _calculate_distance(self, query_emb, support_emb):
        """Negative Euclidean distance with temperature scaling.

        Args:
            query_emb:   [N_q, D]
            support_emb: [N_s, D]
        Returns:
            [N_q, N_s] distance matrix
        """
        return -torch.cdist(query_emb, support_emb) / self.dist_temperature

    def _forward(self, train_x, train_y, test_x, test_y, train_mode):
        """Core forward: project -> distances -> weighted prediction -> loss."""
        if train_mode:
            self.projection.train()
        else:
            self.projection.eval()

        ctx = torch.enable_grad if train_mode else torch.no_grad

        with ctx():
            support_proj = self.projection(train_x)
            query_proj = self.projection(test_x)

            dists = self._calculate_distance(query_proj, support_proj)
            weights = torch.softmax(dists, dim=1)      # [N_q, N_s]
            preds = torch.mm(weights, train_y)          # [N_q, 1]
            loss = F.mse_loss(preds, test_y)

        # Metrics (always no_grad)
        with torch.no_grad():
            preds_np = preds.cpu().numpy()
            targets_np = test_y.cpu().numpy()
            mae = np.abs(preds_np - targets_np).mean()
            rmse = np.sqrt(np.mean((preds_np - targets_np) ** 2))
            r2 = r2_score(targets_np, preds_np) if len(targets_np) > 1 else 0.0

        return mae, loss, rmse, r2, preds_np

    def train_step(self, train_x, train_y, test_x, test_y):
        """One training step for a single task."""
        self.task_counter += 1
        mae, loss, rmse, r2, preds = self._forward(
            train_x, train_y, test_x, test_y, train_mode=True)

        if self.optimizer is not None:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return mae, loss.item(), rmse, r2, preds

    def evaluate(self, train_x, train_y, test_x, test_y):
        """Evaluate on a single task (no gradient)."""
        mae, loss, rmse, r2, preds = self._forward(
            train_x, train_y, test_x, test_y, train_mode=False)
        return mae, loss.item(), rmse, r2, preds

    def save(self, path):
        """Save projection network state."""
        torch.save(self.projection.state_dict(), path)

    def load(self, path):
        """Load projection network state."""
        state = torch.load(path, map_location=self.dev)
        self.projection.load_state_dict(state)
