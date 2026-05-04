# ijepa_spectra2d — Precompute I-JEPA embeddings for downstream tasks
#
# Extracts frozen target_encoder features from precomputed GADF .pt files
# and saves them as emb_supp.pt / emb_query.pt either inside each task
# directory or under a mirrored external root.
#
# Usage:
#   python scripts/precompute_embeddings.py \
#       --data_path data/Soil_NIR_AGG_mixed_gadf2d \
#       --checkpoint logs/gadf_soil_nir/gadf_jepa-latest.pth.tar \
#       --model_name vit_base \
#       --batch_size 64 \
#       --device cuda
#
#   # Save embeddings outside the dataset tree:
#   python scripts/precompute_embeddings.py \
#       --data_path data/SoilDataset_NIR_gadf2d \
#       --checkpoint logs/.../model.pth.tar \
#       --output_root artifacts/embeddings/model_ep700_mean

import argparse
import json
import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main_finetune_gadf2d import load_ijepa2d_encoder


@torch.no_grad()
def extract_embeddings(encoder, images, device, batch_size=64, emb_mode='mean'):
    """Extract I-JEPA embeddings: [N, 1, 224, 224] -> [N, D] or [N, P, D].

    emb_mode='mean':    mean-pool patch tokens → [N, D]
    emb_mode='patches': keep all patch tokens  → [N, P, D]  (P=196 for vit_base/16)
    """
    encoder.eval()
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size].to(device)
        tokens = encoder(batch, masks=None)   # [B, P, D]
        if emb_mode == 'patches':
            emb = tokens                      # [B, P, D]
        else:
            emb = tokens.mean(dim=1)          # [B, D]
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)


def main(args):
    # Load encoder
    encoder, embed_dim = load_ijepa2d_encoder(
        args.checkpoint,
        model_name=args.model_name,
        patch_size=args.patch_size,
        crop_size=args.crop_size,
        in_chans=1,
    )
    encoder = encoder.to(args.device)
    encoder.eval()
    print(f'Embedding dim: {embed_dim}')

    # Load normalization stats
    norm_mean, norm_std = 0.0, 1.0
    if args.norm_stats and os.path.exists(args.norm_stats):
        with open(args.norm_stats) as f:
            stats = json.load(f)
        key = 'with_diagonal' if args.diagonal else 'no_diagonal'
        norm_mean = stats[key]['gadf_mean']
        norm_std = stats[key]['gadf_std']
        print(f'Normalization: mean={norm_mean}, std={norm_std}')

    # Process each task directory
    task_dirs = sorted([
        d for d in os.listdir(args.data_path)
        if os.path.isdir(os.path.join(args.data_path, d))
    ])

    if args.output_root:
        os.makedirs(args.output_root, exist_ok=True)
        meta = {
            'checkpoint': args.checkpoint,
            'data_path': args.data_path,
            'model_name': args.model_name,
            'patch_size': args.patch_size,
            'crop_size': args.crop_size,
            'norm_stats': args.norm_stats,
            'diagonal': args.diagonal,
            'emb_mode': args.emb_mode,
        }
        with open(os.path.join(args.output_root, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    total_samples = 0
    processed = 0

    for task_name in task_dirs:
        task_dir = os.path.join(args.data_path, task_name)

        # Choose filenames based on mode
        suffix = '_patches' if args.emb_mode == 'patches' else ''
        output_task_dir = task_dir
        if args.output_root:
            output_task_dir = os.path.join(args.output_root, task_name)
            os.makedirs(output_task_dir, exist_ok=True)

        emb_supp_path = os.path.join(output_task_dir, f'emb_supp{suffix}.pt')
        emb_query_path = os.path.join(output_task_dir, f'emb_query{suffix}.pt')
        if not args.overwrite and os.path.exists(emb_supp_path) and os.path.exists(emb_query_path):
            processed += 1
            continue

        # Load GADF tensors
        supp_path = os.path.join(task_dir, 'X_supp.pt')
        query_path = os.path.join(task_dir, 'X_query.pt')
        if not os.path.exists(supp_path) or not os.path.exists(query_path):
            print(f'  SKIP {task_name}: missing X_supp.pt or X_query.pt')
            continue

        X_supp = torch.load(supp_path, map_location='cpu').float()
        X_query = torch.load(query_path, map_location='cpu').float()

        # Normalize
        X_supp = (X_supp - norm_mean) / norm_std
        X_query = (X_query - norm_mean) / norm_std

        # Extract embeddings
        emb_supp = extract_embeddings(encoder, X_supp, args.device, args.batch_size, args.emb_mode)
        emb_query = extract_embeddings(encoder, X_query, args.device, args.batch_size, args.emb_mode)

        # Save
        torch.save(emb_supp, emb_supp_path)
        torch.save(emb_query, emb_query_path)

        n = emb_supp.shape[0] + emb_query.shape[0]
        total_samples += n
        processed += 1
        print(f'  [{processed}/{len(task_dirs)}] {task_name}: '
              f'supp={emb_supp.shape}, query={emb_query.shape}')

    print(f'\nDone: {processed} tasks, {total_samples} samples, '
          f'embed_dim={embed_dim}, emb_mode={args.emb_mode}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute I-JEPA embeddings')
    parser.add_argument('--data_path', type=str,
                        default='data/Soil_NIR_AGG_mixed_gadf2d')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to I-JEPA pretrained checkpoint')
    parser.add_argument('--model_name', type=str, default='vit_base')
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--norm_stats', type=str,
                        default='data/gadf_norm_stats.json')
    parser.add_argument('--diagonal', action='store_true',
                        help='Use with_diagonal norm stats')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing embeddings')
    parser.add_argument('--output_root', type=str, default=None,
                        help='Optional external directory where embeddings are '
                             'stored in per-task subfolders')
    parser.add_argument('--emb_mode', type=str, default='mean',
                        choices=['mean', 'patches'],
                        help='mean: pool patch tokens → [N,D]; '
                             'patches: keep all tokens → [N,P,D]')
    main(parser.parse_args())
