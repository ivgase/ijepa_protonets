"""Smoke test for main_finetune_gadf2d.py

Tests the full fine-tuning pipeline with synthetic data:
  1. Spectra → GADF 2D conversion
  2. Encoder (random weights) + regression head
  3. Forward/backward pass
  4. Checkpoint save/load round-trip
  5. Linear probing mode

No GPU, checkpoint, or real data required.
"""

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.models.vision_transformer as vit
from main_finetune_gadf2d import (
    GADF2DForRegression,
    EarlyStop,
    load_ijepa2d_encoder,
    spectra_to_gadf,
    train_one_epoch,
    evaluate,
)
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device('cpu')
print(f"device: {device}")

# ============================================================================
# 1. Test spectra_to_gadf
# ============================================================================
print("\n[1] spectra_to_gadf ...")
n_samples, n_wavelengths = 8, 1050
spectra = np.random.randn(n_samples, n_wavelengths).astype(np.float32)
gadf_imgs = spectra_to_gadf(spectra, image_size=224)
print(f"    spectra: ({n_samples}, {n_wavelengths}) → GADF: {tuple(gadf_imgs.shape)}")
assert gadf_imgs.shape == (n_samples, 1, 224, 224), f"Bad shape: {gadf_imgs.shape}"
assert gadf_imgs.dtype == torch.float32
print("    OK")

# ============================================================================
# 2. Test GADF2DForRegression forward pass
# ============================================================================
print("\n[2] GADF2DForRegression forward ...")
encoder = vit.vit_base(img_size=[224], patch_size=16, in_chans=1)
embed_dim = encoder.embed_dim
model = GADF2DForRegression(encoder, embed_dim, out_dim=1)
print(f"    Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

x = gadf_imgs[:4]  # batch of 4
out = model(x)
print(f"    Input: {tuple(x.shape)} → Output: {tuple(out.shape)}")
assert out.shape == (4, 1), f"Bad output shape: {out.shape}"
print("    OK")

# ============================================================================
# 3. Test backward pass
# ============================================================================
print("\n[3] Backward pass ...")
y = torch.randn(4, 1)
loss = nn.MSELoss()(out, y)
loss.backward()
print(f"    Loss: {loss.item():.6f}")

# Check gradients exist
n_grad = sum(1 for p in model.parameters() if p.grad is not None)
n_total = sum(1 for p in model.parameters())
print(f"    Gradients: {n_grad}/{n_total} params")
assert n_grad > 0, "No gradients computed"
print("    OK")

# ============================================================================
# 4. Test train_one_epoch + evaluate
# ============================================================================
print("\n[4] train_one_epoch + evaluate ...")
model.zero_grad()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5)
loss_fn = nn.MSELoss()

train_x = gadf_imgs
train_y = torch.randn(n_samples, 1)
train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=4, shuffle=True)
val_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=4, shuffle=False)

train_loss, train_mae = train_one_epoch(
    model, train_loader, optimizer, scheduler, loss_fn,
    device, val_loss=float('inf'), scheduler_type='ReduceLROnPlateau')
print(f"    Train loss: {train_loss:.6f}, MAE: {train_mae:.6f}")

val_loss, val_mae, val_rmse = evaluate(model, val_loader, loss_fn, device)
print(f"    Val loss: {val_loss:.6f}, MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
assert np.isfinite(train_loss) and np.isfinite(val_loss), "Non-finite losses"
print("    OK")

# ============================================================================
# 5. Test checkpoint save/load round-trip
# ============================================================================
print("\n[5] Checkpoint round-trip ...")
with tempfile.TemporaryDirectory() as tmpdir:
    # Simulate a pretraining checkpoint (with DDP module. prefix)
    ckpt_path = os.path.join(tmpdir, 'fake_pretrain.pth.tar')
    state_dict = {}
    for k, v in encoder.state_dict().items():
        state_dict[f'module.{k}'] = v
    torch.save({
        'target_encoder': state_dict,
        'epoch': 42,
        'loss': 0.123,
    }, ckpt_path)

    # Load it back
    loaded_encoder, loaded_dim = load_ijepa2d_encoder(
        ckpt_path, model_name='vit_base', patch_size=16, crop_size=224, in_chans=1)
    assert loaded_dim == embed_dim, f"embed_dim mismatch: {loaded_dim} vs {embed_dim}"

    # Verify weights match
    for k in encoder.state_dict():
        orig = encoder.state_dict()[k]
        loaded = loaded_encoder.state_dict()[k]
        assert torch.equal(orig, loaded), f"Weight mismatch at {k}"

    print("    Saved with module. prefix, loaded successfully, weights match")

    # Test EarlyStop save
    es_path = os.path.join(tmpdir, 'best.pth')
    es = EarlyStop(patience=3, mode='min')
    model2 = GADF2DForRegression(loaded_encoder, loaded_dim, out_dim=1)
    es(0.5, model2, es_path, epoch=1)
    assert os.path.exists(es_path), "EarlyStop didn't save"
    assert not es.early_stop

    es(0.6, model2, es_path, epoch=2)
    es(0.7, model2, es_path, epoch=3)
    es(0.8, model2, es_path, epoch=4)
    assert es.early_stop, "EarlyStop should have triggered"
    print("    EarlyStop works correctly")
print("    OK")

# ============================================================================
# 6. Test linear probing mode
# ============================================================================
print("\n[6] Linear probing ...")
encoder_lp = vit.vit_base(img_size=[224], patch_size=16, in_chans=1)
model_lp = GADF2DForRegression(encoder_lp, embed_dim, out_dim=1)
for name, param in model_lp.named_parameters():
    if 'head' not in name:
        param.requires_grad_(False)

trainable = sum(p.numel() for p in model_lp.parameters() if p.requires_grad)
frozen = sum(p.numel() for p in model_lp.parameters() if not p.requires_grad)
print(f"    Trainable: {trainable}, Frozen: {frozen}")
# LayerNorm: weight(768) + bias(768) = 1536;  Linear(768,1): weight(768) + bias(1) = 769
expected_trainable = 2 * embed_dim + embed_dim + 1  # 2305
assert trainable == expected_trainable, \
    f"Expected {expected_trainable} trainable (LN + Linear), got {trainable}"
assert frozen > 0, "Nothing is frozen"

out_lp = model_lp(gadf_imgs[:2])
loss_lp = nn.MSELoss()(out_lp, torch.randn(2, 1))
loss_lp.backward()

encoder_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                    for n, p in model_lp.named_parameters() if 'head' not in n)
head_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                 for n, p in model_lp.named_parameters() if 'head' in n)
assert not encoder_grads, "Encoder should have no gradients in linear probing"
assert head_grads, "Head should have gradients"
print("    Encoder frozen, head trainable")
print("    OK")

# ============================================================================
print("\n" + "=" * 50)
print("All smoke tests passed.")
print("=" * 50)
