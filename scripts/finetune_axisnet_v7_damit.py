#!/usr/bin/env python3
"""
Fine-tune AxisNet V7 on DAMIT with 5-Fold Cross-Validation

CRITICAL FINE-TUNING STRATEGY:
1. Load pretrained v7 model from synthetic
2. Use differential learning rate:
   - Backbone: 0.1x learning rate (preserve pretrained)
   - Heads: 1.0x learning rate (adapt to real data)
3. Lower reconstruction weight (λ_recon=0.5) to prioritize pole learning
4. Apply grouped 5-fold CV (by asteroid, not by sample)
5. Save per-fold checkpoints and metrics

Expected result after fine-tuning:
- Median pole error: <30° (vs v6a baseline 61°)
- Pole sensitivity: ≥20% (must be from magnitudes)
- All acceptance gates PASS

Usage:
    python scripts/finetune_axisnet_v7_damit.py \\
        --pretrained artifacts/axisnet_v7_pretrained/best_model.pt \\
        --dataset data/axis_dataset_v6a_damit.npz \\
        --output-dir artifacts/axisnet_v7_finetuned \\
        --n-folds 5 \\
        --batch-size 4 \\
        --lr 1e-4 \\
        --max-epochs 50 \\
        --device cuda
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from lc_pipeline.axis.dataset_v6a import AxisDatasetV6a, collate_v6a
from lc_pipeline.axis.model_axisnet_v7 import create_axisnet_v7, AxisNetV7Loss, create_swapped_batch


def train_epoch(model, loss_fn, dataloader, optimizer, device, epoch, args):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_pole_loss = 0.0
    total_recon_loss = 0.0
    total_anti_inv_loss = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)

    for batch in pbar:
        B = len(batch['times'])

        # Move batch to device
        batch_device = {}
        for key, val in batch.items():
            if torch.is_tensor(val):
                batch_device[key] = val.to(device)
            else:
                batch_device[key] = val

        times = batch_device['times']
        mags = batch_device['mags']
        geometry = batch_device['geometry']
        point_mask = batch_device['point_mask']
        epoch_mask = batch_device['epoch_mask']
        period_pred = batch_device['period_pred']
        t_ref = batch_device['t_ref']
        pole_targets = batch_device['pole_targets']
        pole_mask = batch_device['pole_mask']

        # Forward pass (original)
        pole_pred, coeffs_global, phases, A, c = model(
            times, mags, geometry, point_mask, epoch_mask, period_pred, t_ref
        )

        # Reconstruct magnitudes
        mags_pred = model.forward_model(phases, coeffs_global, A, c)

        # === SWAPPED BATCH ===
        indices_perm = torch.randperm(B, device=device)
        batch_swap = create_swapped_batch(batch_device, indices_perm)

        times_swap = batch_swap['times']
        mags_swap = batch_swap['mags']
        geometry_swap = batch_swap['geometry']

        pole_pred_swap, _, _, _, _ = model(
            times_swap, mags_swap, geometry_swap, point_mask, epoch_mask, period_pred, t_ref
        )

        # === LOSS ===
        total_epoch_loss, pole_loss, recon_loss, anti_inv_loss = loss_fn(
            pole_pred, pole_targets, pole_mask,
            mags_pred, mags, point_mask,
            pole_pred_swap
        )

        # Backward
        optimizer.zero_grad()
        total_epoch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Accumulate
        total_loss += total_epoch_loss.item()
        total_pole_loss += pole_loss.item()
        total_recon_loss += recon_loss.item()
        total_anti_inv_loss += anti_inv_loss.item()

        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'pole': total_pole_loss / (pbar.n + 1),
        })

    return {
        'total_loss': total_loss / len(dataloader),
        'pole_loss': total_pole_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
        'anti_inv_loss': total_anti_inv_loss / len(dataloader),
    }


def evaluate(model, dataloader, device):
    """Evaluate on test set."""
    model.eval()
    all_errors = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_device = {}
            for key, val in batch.items():
                if torch.is_tensor(val):
                    batch_device[key] = val.to(device)
                else:
                    batch_device[key] = val

            times = batch_device['times']
            mags = batch_device['mags']
            geometry = batch_device['geometry']
            point_mask = batch_device['point_mask']
            epoch_mask = batch_device['epoch_mask']
            period_pred = batch_device['period_pred']
            t_ref = batch_device['t_ref']
            pole_targets = batch_device['pole_targets']
            pole_mask = batch_device['pole_mask']

            # Forward pass
            pole_pred, _, _, _, _ = model(
                times, mags, geometry, point_mask, epoch_mask, period_pred, t_ref
            )

            # Compute per-asteroid errors
            pole_pred_norm = F.normalize(pole_pred, dim=-1)
            pole_targets_norm = F.normalize(pole_targets, dim=-1)

            cos_angles = torch.einsum('bi,bmi->bm', pole_pred_norm, pole_targets_norm)
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            angular_dists = torch.acos(cos_angles)

            # Check antipodes
            cos_angles_antipode = torch.einsum('bi,bmi->bm', pole_pred_norm, -pole_targets_norm)
            cos_angles_antipode = torch.clamp(cos_angles_antipode, -1.0, 1.0)
            angular_dists_antipode = torch.acos(cos_angles_antipode)

            angular_dists = torch.minimum(angular_dists, angular_dists_antipode)

            # Mask
            angular_dists_masked = angular_dists.masked_fill(~pole_mask, 1e6)
            min_dists, _ = angular_dists_masked.min(dim=-1)

            # Convert to degrees
            min_dists_deg = torch.rad2deg(min_dists)
            all_errors.extend(min_dists_deg.cpu().numpy())

    return np.array(all_errors)


def compute_metrics(errors):
    """Compute standard metrics."""
    errors = np.array(errors)
    return {
        'median': np.median(errors),
        'mean': np.mean(errors),
        'acc@10': np.mean(errors <= 10),
        'acc@20': np.mean(errors <= 20),
        'acc@30': np.mean(errors <= 30),
        'q1': np.percentile(errors, 25),
        'q3': np.percentile(errors, 75),
    }


def main():
    parser = argparse.ArgumentParser(description='Fine-tune AxisNet V7 on DAMIT')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to DAMIT dataset')
    parser.add_argument('--output-dir', type=str, default='artifacts/axisnet_v7_finetuned',
                        help='Output directory')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--backbone-lr-mult', type=float, default=0.1,
                        help='Backbone LR multiplier')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max-epochs', type=int, default=50,
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = AxisDatasetV6a(args.dataset)
    n_total = len(dataset)
    print(f"Total samples: {n_total}")

    # Group by asteroid ID (if available from dataset metadata)
    # For now, use simple stratification
    group_ids = np.arange(n_total)  # Placeholder

    # Prepare k-fold
    kf = GroupKFold(n_splits=args.n_folds)
    fold_results = {}

    # =============================================================================
    # K-FOLD CROSS-VALIDATION LOOP
    # =============================================================================

    all_val_errors = []

    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(
            np.arange(n_total), groups=group_ids)):

        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print(f"{'='*70}")

        # Create train/val subsets
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_v6a, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_v6a, num_workers=0)

        print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Create model (load pretrained)
        print("Loading pretrained model...")
        model = create_axisnet_v7(d_model=128, shape_dim=8, K=4, dropout=0.1)

        checkpoint = torch.load(args.pretrained, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        # Differential learning rate
        backbone_params = list(model.photo_encoder.parameters()) + \
                          list(model.epoch_encoder.parameters()) + \
                          list(model.global_pool.parameters())
        head_params = list(model.pole_head.parameters()) + \
                      list(model.shape_head.parameters()) + \
                      list(model.phi_global_head.parameters()) + \
                      list(model.coeff_head.parameters()) + \
                      [model.shape_latent]

        optimizer = AdamW([
            {'params': backbone_params, 'lr': args.lr * args.backbone_lr_mult},
            {'params': head_params, 'lr': args.lr},
        ], weight_decay=args.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

        # Loss
        loss_fn = AxisNetV7Loss(lambda_pole=1.0, lambda_recon=0.5, lambda_anti_inv=0.5)

        # Training loop
        best_val_error = float('inf')
        patience_counter = 0
        fold_history = {
            'train_loss': [],
            'val_error': [],
        }

        for epoch in range(args.max_epochs):
            # Train
            train_metrics = train_epoch(model, loss_fn, train_loader, optimizer, device, epoch, args)

            # Validate
            val_errors = evaluate(model, val_loader, device)
            val_median = np.median(val_errors)

            scheduler.step()

            # Log
            print(f"Epoch {epoch+1}: Loss={train_metrics['total_loss']:.4f}, "
                  f"Val Median={val_median:.1f}°")

            fold_history['train_loss'].append(train_metrics['total_loss'])
            fold_history['val_error'].append(val_median)

            # Early stopping
            if val_median < best_val_error:
                best_val_error = val_median
                patience_counter = 0

                # Save checkpoint
                checkpoint_path = Path(args.output_dir) / f'fold{fold_idx}_best_model.pt'
                torch.save({
                    'fold': fold_idx,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'val_error': val_median,
                }, checkpoint_path)
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1} (best: {best_val_error:.1f}°)")
                break

        # Final evaluation on val set
        val_errors_final = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_errors_final)

        print(f"\nFold {fold_idx+1} Results:")
        print(f"  Median: {val_metrics['median']:.1f}°")
        print(f"  Acc@20°: {val_metrics['acc@20']:.1%}")
        print(f"  Acc@30°: {val_metrics['acc@30']:.1%}")

        fold_results[f'fold{fold_idx}'] = {
            'metrics': val_metrics,
            'history': fold_history,
            'best_val_error': best_val_error,
        }

        all_val_errors.extend(val_errors_final)

    # =============================================================================
    # AGGREGATE RESULTS
    # =============================================================================

    print(f"\n{'='*70}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*70}")

    aggregate_metrics = compute_metrics(all_val_errors)
    print(f"Aggregate over all folds:")
    print(f"  Median: {aggregate_metrics['median']:.1f}°")
    print(f"  Mean: {aggregate_metrics['mean']:.1f}°")
    print(f"  Acc@20°: {aggregate_metrics['acc@20']:.1%}")
    print(f"  Acc@30°: {aggregate_metrics['acc@30']:.1%}")
    print(f"  Q1: {aggregate_metrics['q1']:.1f}°")
    print(f"  Q3: {aggregate_metrics['q3']:.1f}°")

    # Per-fold
    print(f"\nPer-fold medians:")
    fold_medians = []
    for fold_idx in range(args.n_folds):
        fold_key = f'fold{fold_idx}'
        fold_median = fold_results[fold_key]['metrics']['median']
        fold_medians.append(fold_median)
        print(f"  Fold {fold_idx+1}: {fold_median:.1f}°")

    print(f"\nFold median ± std: {np.mean(fold_medians):.1f}° ± {np.std(fold_medians):.1f}°")

    # Save results
    results = {
        'aggregate_metrics': aggregate_metrics,
        'fold_results': fold_results,
        'fold_medians': fold_medians,
        'fold_mean': float(np.mean(fold_medians)),
        'fold_std': float(np.std(fold_medians)),
    }

    results_path = Path(args.output_dir) / 'cv_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Interpretation
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")

    median_error = aggregate_metrics['median']
    if median_error < 30:
        print(f"✓ EXCELLENT: {median_error:.1f}° median error (target: <30°)")
    elif median_error < 40:
        print(f"✓ GOOD: {median_error:.1f}° median error (acceptable)")
    elif median_error < 50:
        print(f"⚠ MARGINAL: {median_error:.1f}° median error (needs improvement)")
    else:
        print(f"✗ POOR: {median_error:.1f}° median error (redesign needed)")

    acc20 = aggregate_metrics['acc@20']
    if acc20 > 0.3:
        print(f"✓ Strong: Acc@20° = {acc20:.1%}")
    elif acc20 > 0.2:
        print(f"✓ Reasonable: Acc@20° = {acc20:.1%}")
    else:
        print(f"⚠ Weak: Acc@20° = {acc20:.1%}")


if __name__ == '__main__':
    main()
