#!/usr/bin/env python3
"""
Train AxisNet V7 on Synthetic Data with Anti-Invariance Loss

CRITICAL V7 DESIGN ENFORCEMENTS:
1. Pole head receives PHOTOMETRY ONLY (no geometry)
2. Global harmonic template (no per-epoch coefficients)
3. Per-epoch modulation via pole-dependent dot products
4. Anti-invariance loss: explicitly penalizes magnitude-ignoring
5. Swapped-batch augmentation during training

Loss schedule:
    Epochs 0-9:   λ_recon = 0.0  (pole-only warmup)
    Epochs 10-19: λ_recon ramps 0 → 1.0
    Epochs 20+:   λ_recon = 1.0, anti_inv = 0.5 (full v7 loss)

Expected result after 100 epochs:
- Median pole error: 30-45° (significant improvement over v6a's 61°)
- Pole sensitivity: ≥20% (must be from magnitudes, not geometry)
- Mag-swap test: >10° degradation (confirms photometry use)

Usage:
    python scripts/train_axisnet_v7.py \\
        --dataset data/synth_axis_v6a_100k.npz \\
        --output-dir artifacts/axisnet_v7_pretrained \\
        --batch-size 8 \\
        --lr 1e-3 \\
        --max-epochs 100 \\
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from lc_pipeline.axis.dataset_v6a import AxisDatasetV6a, collate_v6a
from lc_pipeline.axis.model_axisnet_v7 import create_axisnet_v7, AxisNetV7Loss, create_swapped_batch


def train_epoch(model, loss_fn, dataloader, optimizer, device, epoch, args):
    """
    Train one epoch with anti-invariance loss.
    """
    model.train()
    total_loss = 0.0
    total_pole_loss = 0.0
    total_recon_loss = 0.0
    total_anti_inv_loss = 0.0

    # Update loss schedule
    if hasattr(loss_fn, 'set_epoch'):
        loss_fn.set_epoch(epoch)

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

        # === ORIGINAL BATCH ===
        times = batch_device['times']  # (B, E, P)
        mags = batch_device['mags']    # (B, E, P)
        geometry = batch_device['geometry']  # (B, E, 6)
        point_mask = batch_device['point_mask']  # (B, E, P)
        epoch_mask = batch_device['epoch_mask']  # (B, E)
        period_pred = batch_device['period_pred']  # (B,)
        t_ref = batch_device['t_ref']  # (B,)
        pole_targets = batch_device['pole_targets']  # (B, max_mirrors, 3)
        pole_mask = batch_device['pole_mask']  # (B, max_mirrors)

        # Forward pass (original)
        pole_pred, coeffs_global, phases, A, c = model(
            times, mags, geometry, point_mask, epoch_mask, period_pred, t_ref
        )

        # Reconstruct magnitudes
        mags_pred = model.forward_model(phases, coeffs_global, A, c)

        # === SWAPPED BATCH (for anti-invariance) ===
        # Create magnitude permutation
        indices_perm = torch.randperm(B, device=device)
        batch_swap = create_swapped_batch(batch_device, indices_perm)

        # Forward pass (swapped)
        times_swap = batch_swap['times']
        mags_swap = batch_swap['mags']
        geometry_swap = batch_swap['geometry']

        pole_pred_swap, _, phases_swap, A_swap, c_swap = model(
            times_swap, mags_swap, geometry_swap, point_mask, epoch_mask, period_pred, t_ref
        )

        # === LOSS COMPUTATION ===
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

        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (pbar.n + 1),
            'pole': total_pole_loss / (pbar.n + 1),
            'recon': total_recon_loss / (pbar.n + 1),
            'anti_inv': total_anti_inv_loss / (pbar.n + 1),
        })

    return {
        'total_loss': total_loss / len(dataloader),
        'pole_loss': total_pole_loss / len(dataloader),
        'recon_loss': total_recon_loss / len(dataloader),
        'anti_inv_loss': total_anti_inv_loss / len(dataloader),
    }


def evaluate(model, dataloader, device):
    """
    Evaluate model on validation set.

    Returns:
        errors_deg: (N,) per-asteroid angular errors in degrees
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            B = len(batch['times'])

            # Move to device
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
            pole_targets = batch_device['pole_targets']  # (B, max_mirrors, 3)
            pole_mask = batch_device['pole_mask']  # (B, max_mirrors)

            # Forward pass
            pole_pred, _, _, _, _ = model(
                times, mags, geometry, point_mask, epoch_mask, period_pred, t_ref
            )

            # Compute per-asteroid errors
            pole_pred_norm = F.normalize(pole_pred, dim=-1)  # (B, 3)
            pole_targets_norm = F.normalize(pole_targets, dim=-1)  # (B, max_mirrors, 3)

            cos_angles = torch.einsum('bi,bmi->bm', pole_pred_norm, pole_targets_norm)
            cos_angles = torch.clamp(cos_angles, -1.0, 1.0)
            angular_dists = torch.acos(cos_angles)  # (B, max_mirrors)

            # Check antipodes
            cos_angles_antipode = torch.einsum('bi,bmi->bm', pole_pred_norm, -pole_targets_norm)
            cos_angles_antipode = torch.clamp(cos_angles_antipode, -1.0, 1.0)
            angular_dists_antipode = torch.acos(cos_angles_antipode)

            angular_dists = torch.minimum(angular_dists, angular_dists_antipode)

            # Mask invalid mirrors
            angular_dists_masked = angular_dists.masked_fill(~pole_mask, 1e6)
            min_dists, _ = angular_dists_masked.min(dim=-1)

            # Convert to degrees
            min_dists_deg = torch.rad2deg(min_dists)
            all_errors.extend(min_dists_deg.cpu().numpy())

    return np.array(all_errors)


def main():
    parser = argparse.ArgumentParser(description='Train AxisNet V7')
    parser.add_argument('--dataset', type=str, default='data/synth_axis_v6a_100k.npz',
                        help='Path to synthetic dataset')
    parser.add_argument('--output-dir', type=str, default='artifacts/axisnet_v7_pretrained',
                        help='Output directory for checkpoints')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--max-epochs', type=int, default=100,
                        help='Maximum epochs')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
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

    # Split into train/val (80/20)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_v6a, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_v6a, num_workers=0)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    print("Creating AxisNet v7...")
    model = create_axisnet_v7(d_model=128, shape_dim=8, K=4, dropout=0.1)
    model = model.to(device)

    # Loss and optimizer
    loss_fn = AxisNetV7Loss(lambda_pole=1.0, lambda_recon=1.0, lambda_anti_inv=0.5)

    # Add epoch scheduling to loss function
    loss_fn.set_epoch = lambda epoch: None  # Placeholder, will override

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

    # Training loop
    best_val_error = float('inf')
    patience_counter = 0

    history = {
        'train_loss': [],
        'val_error': [],
        'pole_loss': [],
        'recon_loss': [],
        'anti_inv_loss': [],
    }

    for epoch in range(args.max_epochs):
        # Update loss schedule
        if epoch < 10:
            loss_fn.lambda_recon_current = 0.0
        elif epoch < 20:
            progress = (epoch - 10) / 10.0
            loss_fn.lambda_recon_current = 1.0 * progress
        else:
            loss_fn.lambda_recon_current = 1.0

        # Train
        train_metrics = train_epoch(model, loss_fn, train_loader, optimizer, device, epoch, args)

        # Validate
        val_errors = evaluate(model, val_loader, device)
        val_median = np.median(val_errors)

        # Learning rate update
        scheduler.step()

        # Log
        print(f"Epoch {epoch+1}: Train Loss={train_metrics['total_loss']:.4f}, "
              f"Pole={train_metrics['pole_loss']:.4f}, "
              f"Recon={train_metrics['recon_loss']:.4f} (λ={loss_fn.lambda_recon_current:.2f}), "
              f"AntiInv={train_metrics['anti_inv_loss']:.4f}")
        print(f"  Val Median Error: {val_median:.1f}°")

        history['train_loss'].append(train_metrics['total_loss'])
        history['val_error'].append(val_median)
        history['pole_loss'].append(train_metrics['pole_loss'])
        history['recon_loss'].append(train_metrics['recon_loss'])
        history['anti_inv_loss'].append(train_metrics['anti_inv_loss'])

        # Early stopping
        if val_median < best_val_error:
            best_val_error = val_median
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = Path(args.output_dir) / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_error': val_median,
                'history': history,
            }, checkpoint_path)
            print(f"  → Saved best model: {checkpoint_path}")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch+1} (best: {best_val_error:.1f}°)")
            break

    # Save final history
    history_path = Path(args.output_dir) / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best median error: {best_val_error:.1f}°")
    print(f"Checkpoints saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
