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


# ============================================================================
# Convolutional encoder: ResNet1D
# Ported from fewshot_nir_orig/models/cnn.py (ResNet1D / ResidualBlock1D)
# without the MAML vars pattern and without the final linear prediction head.
# ============================================================================

class ResidualBlock1D(nn.Module):
    """Basic residual block for 1D convolutions."""

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class EmbResNet1D(nn.Module):
    """ResNet1D encoder adapted for I-JEPA embeddings.

    Ported from fewshot_nir_orig/models/cnn.py (ResNet1D, ResNet18 config:
    layers=[2,2,2,2], channels=[64,128,256,512]) without the MAML vars pattern
    and without the final prediction head.

    Two input modes (selected by in_channels):
      mean_pool  (in_channels=1):   x [B, D]       → unsqueeze(1) → [B, 1, D]
      patches    (in_channels=196): x [B, 196, D]  → used as-is

    Output: [B, output_dim]
    """

    def __init__(self, in_channels=1, output_dim=512):
        super().__init__()
        self._in_ch = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        # Optional head if output_dim != 512
        self.head = nn.Linear(512, output_dim) if output_dim != 512 else None

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self._in_ch != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self._in_ch, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        layers = [ResidualBlock1D(self._in_ch, out_channels, stride, downsample)]
        self._in_ch = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:          # [B, D] mean_pool mode
            x = x.unsqueeze(1)    # → [B, 1, D]
        # x is now [B, C, L]
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)  # [B, 512]
        if self.head is not None:
            x = self.head(x)
        return x


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


class TransformerAggregator(nn.Module):
    """Small transformer that aggregates ViT patch tokens into a single embedding.

    Flow:
        [B, N, input_dim]
          → Linear(input_dim, d_model)          # compress before attention
          → prepend CLS token + learnable pos_embed
          → L × Block(d_model, num_heads, ...)   # self-attention over patches
          → extract CLS (position 0)
          → LayerNorm → Linear(d_model, output_dim)
          → [B, output_dim]

    The CLS token learns a selective readout over the patch sequence, which is
    more expressive than mean-pooling — especially useful for few-shot regression
    where certain spatial regions (patches) may be more task-relevant than others.
    """

    def __init__(self, input_dim=768, d_model=256, output_dim=128,
                 num_patches=196, num_layers=1, num_heads=4,
                 mlp_ratio=2.0, dropout=0.0, attn_dropout=0.0):
        # num_patches kept for API compatibility but no longer used (pos_embed removed)
        super().__init__()
        from src.models.vision_transformer import Block
        from src.utils.tensors import trunc_normal_

        self.input_proj = nn.Linear(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.blocks = nn.ModuleList([
            Block(dim=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop=dropout, attn_drop=attn_dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, output_dim)

        # Initialize weights
        trunc_normal_(self.cls_token, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        """
        Args:
            x: [B, N, input_dim]  patch tokens from ViT encoder
        Returns:
            [B, output_dim]
        """
        B = x.shape[0]
        x = self.input_proj(x)                                    # [B, N, d_model]
        cls = self.cls_token.expand(B, -1, -1)                    # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)                            # [B, N+1, d_model]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x[:, 0])                                    # CLS token → [B, d_model]
        return self.output_proj(x)                                 # [B, output_dim]


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
