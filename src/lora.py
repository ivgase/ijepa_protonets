"""
LoRA (Low-Rank Adaptation) implementation for ijepa_spectra2d.

Based on the paper: "LoRA: Low-Rank Adaptation of Large Language Models"
https://arxiv.org/abs/2106.09685

Adapted from SpectraI-JEPA/model/lora.py with:
- target_modules changed to ['attn.qkv', 'attn.proj'] for this ViT
- Fix: collect replacements before applying (safe iteration)
- Fix: create LoRA params without explicit device (moved by model.to())
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class LoRALayer(nn.Module):
    """
    Base LoRA layer: low-rank adaptation matrices A and B.

    Computes: h = (alpha/r) * (dropout(x) @ A^T @ B^T)
    where A is [r, in_features] and B is [out_features, r].
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adaptation.
    Keeps original weights frozen, adds trainable low-rank matrices.
    """
    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            r=r, alpha=alpha, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias


def apply_lora(
    model: nn.Module,
    target_modules: Optional[list] = None,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> int:
    """
    Apply LoRA to nn.Linear modules whose name contains any of the target strings.

    Default targets: ['attn.qkv', 'attn.proj'] (attention layers in this ViT).

    Returns the number of modules replaced.
    """
    if target_modules is None:
        target_modules = ['attn.qkv', 'attn.proj']

    # Collect replacements first to avoid modifying model during iteration
    replacements = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not any(target in name for target in target_modules):
            continue

        parts = name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = name

        replacements.append((parent, attr_name, module))

    # Apply replacements
    for parent, attr_name, linear in replacements:
        lora_linear = LoRALinear(linear, r=r, alpha=alpha, dropout=dropout)
        setattr(parent, attr_name, lora_linear)

    return len(replacements)


def count_trainable_parameters(model: nn.Module) -> tuple:
    """Returns (trainable_params, total_params, percentage)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total if total > 0 else 0
    return trainable, total, percentage


def merge_lora_weights(model: nn.Module) -> None:
    """
    Merge LoRA weights into base Linear layers (irreversible).
    W' = W + B @ A * (alpha / r)
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta_weight = (module.lora.lora_B @ module.lora.lora_A) * module.lora.scaling
                module.linear.weight.add_(delta_weight)
