"""Forward pass smoke test for ijepa_spectra2d.

Runs the full pretraining forward pass with synthetic data (no HDF5 required).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import math

import torch
import torch.nn.functional as F

from src.helper import init_model
from src.masks.multiblock import MaskCollator
from src.masks.utils import apply_masks
from src.sigreg import SigRegHookCollector, weak_sigreg_loss
from src.utils.tensors import repeat_interleave_batch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE  = 4
IN_CHANS    = 1
CROP_SIZE   = 224
PATCH_SIZE  = 16
NENC        = 1
NPRED       = 4
PRED_DEPTH  = 6
PRED_EMB    = 384   # must be divisible by vit_base num_heads (12)
N_PATCHES   = (CROP_SIZE // PATCH_SIZE) ** 2   # 196

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

# ---------------------------------------------------------------------------
# 1. Synthetic batch
# ---------------------------------------------------------------------------
imgs = torch.randn(BATCH_SIZE, IN_CHANS, CROP_SIZE, CROP_SIZE, device=device)
print(f"\n[1] imgs:          {tuple(imgs.shape)}")

# ---------------------------------------------------------------------------
# 2. Models
# ---------------------------------------------------------------------------
encoder, predictor = init_model(
    device=device,
    patch_size=PATCH_SIZE,
    crop_size=CROP_SIZE,
    pred_depth=PRED_DEPTH,
    pred_emb_dim=PRED_EMB,
    model_name='vit_base',
    in_chans=IN_CHANS,
)
target_encoder = copy.deepcopy(encoder)
for p in target_encoder.parameters():
    p.requires_grad = False

embed_dim = encoder.embed_dim
print(f"\n[2] vit_base  embed_dim={embed_dim}  n_patches={N_PATCHES}")

# ---------------------------------------------------------------------------
# 3. MaskCollator
# ---------------------------------------------------------------------------
collator = MaskCollator(
    input_size=(CROP_SIZE, CROP_SIZE),
    patch_size=PATCH_SIZE,
    pred_mask_scale=(0.15, 0.2),
    enc_mask_scale=(0.85, 1.0),
    aspect_ratio=(0.75, 1.5),
    nenc=NENC,
    npred=NPRED,
    min_keep=10,
    allow_overlap=False,
    symmetric_masking=True,
)

# ---------------------------------------------------------------------------
# 4. Generate masks from a synthetic batch (CPU items, collator runs on CPU)
# ---------------------------------------------------------------------------
synthetic_batch = [(torch.zeros(IN_CHANS, CROP_SIZE, CROP_SIZE), 0)
                   for _ in range(BATCH_SIZE)]
collated_batch, masks_enc, masks_pred = collator(synthetic_batch)

# Move to device
masks_enc  = [m.to(device) for m in masks_enc]
masks_pred = [m.to(device) for m in masks_pred]

n_enc  = masks_enc[0].shape[1]
n_pred = masks_pred[0].shape[1]
print(f"\n[4] masks_enc:  {len(masks_enc)} mask(s)  each {tuple(masks_enc[0].shape)}"
      f"  (n_enc={n_enc})")
print(f"    masks_pred: {len(masks_pred)} mask(s)  each {tuple(masks_pred[0].shape)}"
      f"  (n_pred={n_pred})")

# ---------------------------------------------------------------------------
# 5. Full forward pass (mirrors train.py)
# ---------------------------------------------------------------------------
with torch.no_grad():
    # --- target encoder ---
    h = target_encoder(imgs)                              # [B, N, D]
    print(f"\n[5] target_encoder → h:     {tuple(h.shape)}")

    h = F.layer_norm(h, (h.size(-1),))
    B = len(h)

    h = apply_masks(h, masks_pred)                        # [B*npred, n_pred, D]
    print(f"    apply_masks(pred) → h:  {tuple(h.shape)}")

    h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
    print(f"    repeat_interleave → h:  {tuple(h.shape)}")

    # --- context encoder + predictor ---
    z = encoder(imgs, masks_enc)                          # [B*nenc, n_enc, D]
    print(f"    encoder → z:            {tuple(z.shape)}")

    z = predictor(z, masks_enc, masks_pred)               # matches h
    print(f"    predictor → z:          {tuple(z.shape)}")

    # --- loss ---
    loss = F.smooth_l1_loss(z, h)
    print(f"\n[5] loss: {loss.item():.6f}")

# ---------------------------------------------------------------------------
# 6. Sanity checks
# ---------------------------------------------------------------------------
assert z.shape == h.shape, \
    f"Shape mismatch: predictions {z.shape} vs targets {h.shape}"

expected_seq = B * NPRED * NENC
assert z.shape[0] == expected_seq, \
    f"Expected batch dim {expected_seq}, got {z.shape[0]}"

assert z.shape[2] == embed_dim, \
    f"Expected feature dim {embed_dim}, got {z.shape[2]}"

assert math.isfinite(loss.item()), \
    f"Loss is not finite: {loss.item()}"

# ---------------------------------------------------------------------------
# 7. Weak-SIGReg smoke test on context encoder block activations
# ---------------------------------------------------------------------------
encoder.zero_grad(set_to_none=True)
sigreg_collector = SigRegHookCollector(
    encoder,
    sketch_dim=64,
    representation='mean_tokens',
)
with sigreg_collector.capture():
    _ = encoder(imgs, masks_enc)
loss_sigreg = sigreg_collector.loss()
sigreg_collector.close()

print(f"\n[7] weak-sigreg loss: {loss_sigreg.item():.6f}")

assert math.isfinite(loss_sigreg.item()), \
    f"SIGReg loss is not finite: {loss_sigreg.item()}"

loss_sigreg.backward()
has_sigreg_grad = any(
    p.grad is not None and torch.isfinite(p.grad).all()
    for p in encoder.parameters()
)
assert has_sigreg_grad, "SIGReg backward produced no finite encoder gradients"

direct_sigreg = weak_sigreg_loss(torch.randn(8, embed_dim, device=device), 64)
assert math.isfinite(direct_sigreg.item()), \
    f"Direct SIGReg loss is not finite: {direct_sigreg.item()}"

print("\nAll checks passed.")
