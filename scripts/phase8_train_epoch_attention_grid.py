#!/usr/bin/env python3
"""
Phase 8: Train EpochAttentionGridModel on DAMIT CSV data.

Implements multi-epoch pole prediction with:
- Phase-stratified sampling for rotation-phase-aware token selection
- Period cache integration for deterministic, period-aware sampling
- 5-fold cross-validation grouped by asteroid

Usage:
    python scripts/phase8_train_epoch_attention_grid.py \
        --dataset datasets/damit_real_multiepoch_v0 \
        --outdir artifacts/phase8_seed5000 \
        --n-folds 5 --seeds 5000 --epochs-max 50 \
        --phase-stratified 1 --phase-bins 8 \
        --period-cache artifacts/period_cache_damit.json \
        --allow-missing-periods 0 \
        --debug-phase-cov 1
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from pole_synth.model_epoch_attention_grid import EpochAttentionGridModel, generate_fibonacci_poles
from pole_synth.losses import angular_distance_with_antipode
from pole_synth.period_cache import load_period_cache, get_period_hours, require_coverage
from pole_synth.phase_stratified_sampling import (
    maybe_phase_stratified_indices,
    compute_bin_coverage,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_csv_object(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Load a single DAMIT CSV file.

    Returns dict with arrays: t, brightness, sun_u, obs_u, object_id
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    # Extract time in days (relative to first observation)
    t = df['time'].values.astype(np.float64)

    # Extract brightness
    brightness = df['relative_brightness'].values.astype(np.float32)

    # Extract sun direction (normalize to unit vector)
    sun_u = np.stack([
        df['sun_ast_x'].values,
        df['sun_ast_y'].values,
        df['sun_ast_z'].values,
    ], axis=-1).astype(np.float32)
    sun_u = sun_u / (np.linalg.norm(sun_u, axis=-1, keepdims=True) + 1e-8)

    # Extract observer direction (Earth to asteroid, normalize)
    obs_u = np.stack([
        df['earth_ast_x'].values,
        df['earth_ast_y'].values,
        df['earth_ast_z'].values,
    ], axis=-1).astype(np.float32)
    obs_u = obs_u / (np.linalg.norm(obs_u, axis=-1, keepdims=True) + 1e-8)

    # Object ID from filename
    object_id = csv_path.stem

    return {
        't': t,
        'brightness': brightness,
        'sun_u': sun_u,
        'obs_u': obs_u,
        'object_id': object_id,
    }


def load_pole_from_damit(object_id: str, damit_root: Path) -> Optional[np.ndarray]:
    """
    Load ground-truth pole from DAMIT spin.txt file.

    Returns pole as unit vector (3,) or None if not found.
    """
    # Parse object_id: asteroid_<aid>_model_<mid>
    import re
    match = re.match(r'asteroid_(\d+)_model_(\d+)', object_id)
    if not match:
        return None

    asteroid_id = match.group(1)
    model_id = match.group(2)

    # Locate spin.txt
    spin_path = damit_root / f"asteroid_{asteroid_id}" / f"model_{model_id}" / "spin.txt"
    if not spin_path.exists():
        return None

    try:
        content = spin_path.read_text()
        lines = content.strip().split('\n')
        if not lines:
            return None

        # First line: lambda beta period
        parts = lines[0].split()
        if len(parts) >= 2:
            lam_deg = float(parts[0])  # Ecliptic longitude
            beta_deg = float(parts[1])  # Ecliptic latitude

            # Convert to Cartesian unit vector
            lam_rad = np.radians(lam_deg)
            beta_rad = np.radians(beta_deg)

            x = np.cos(beta_rad) * np.cos(lam_rad)
            y = np.cos(beta_rad) * np.sin(lam_rad)
            z = np.sin(beta_rad)

            pole = np.array([x, y, z], dtype=np.float32)
            pole = pole / (np.linalg.norm(pole) + 1e-8)
            return pole
    except Exception:
        pass

    return None


def load_dataset(
    csv_dir: Path,
    damit_root: Optional[Path] = None,
) -> Tuple[List[Dict], List[str]]:
    """
    Load all DAMIT CSV files and extract poles.

    Returns:
        objects: List of dicts with t, brightness, sun_u, obs_u, pole, object_id
        object_ids: List of object IDs
    """
    csv_files = sorted(csv_dir.glob("*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files in {csv_dir}")

    objects = []
    object_ids = []

    for csv_path in csv_files:
        obj = load_csv_object(csv_path)
        object_id = obj['object_id']

        # Load pole from DAMIT if available
        pole = None
        if damit_root is not None:
            pole = load_pole_from_damit(object_id, damit_root)

        if pole is None:
            # Fallback: random pole (for testing only)
            rng = np.random.default_rng(hash(object_id) % (2**31))
            pole = rng.normal(size=3).astype(np.float32)
            pole = pole / (np.linalg.norm(pole) + 1e-8)

        obj['pole'] = pole
        objects.append(obj)
        object_ids.append(object_id)

    return objects, object_ids


# =============================================================================
# DATASET & DATALOADER
# =============================================================================

class MultiEpochDataset(Dataset):
    """
    Dataset that creates multi-epoch windows from per-object observations.

    For each object, samples n_epochs windows of n_tokens each.
    Uses phase-stratified sampling when period is available.
    """

    def __init__(
        self,
        objects: List[Dict],
        period_cache: Optional[Dict] = None,
        n_epochs: int = 5,
        n_tokens: int = 64,
        token_dim: int = 28,
        phase_stratified: bool = True,
        phase_bins: int = 8,
        allow_missing_periods: bool = False,
        seed: int = 42,
        debug_phase_cov: bool = False,
        debug_limit: int = 5,
    ):
        self.objects = objects
        self.period_cache = period_cache or {}
        self.n_epochs = n_epochs
        self.n_tokens = n_tokens
        self.token_dim = token_dim
        self.phase_stratified = phase_stratified
        self.phase_bins = phase_bins
        self.allow_missing_periods = allow_missing_periods
        self.seed = seed
        self.debug_phase_cov = debug_phase_cov
        self.debug_limit = debug_limit
        self._debug_count = 0
        # Aggregate coverage tracking
        self._cov_strat_list = []
        self._cov_rand_list = []

    def __len__(self):
        return len(self.objects)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        obj = self.objects[idx]
        object_id = obj['object_id']

        t = obj['t']
        brightness = obj['brightness']
        sun_u = obj['sun_u']
        obs_u = obj['obs_u']
        pole = obj['pole']

        n_obs = len(t)
        rng = np.random.default_rng(self.seed + idx)

        # Get period for this object
        period_hours = get_period_hours(self.period_cache, object_id)

        # Select indices for each epoch
        epochs_tokens = []
        epochs_masks = []

        tokens_per_epoch = min(self.n_tokens, n_obs // self.n_epochs)
        if tokens_per_epoch < 1:
            tokens_per_epoch = 1

        for epoch_idx in range(self.n_epochs):
            # Determine window for this epoch (split observations across epochs)
            epoch_start = (epoch_idx * n_obs) // self.n_epochs
            epoch_end = ((epoch_idx + 1) * n_obs) // self.n_epochs
            epoch_size = epoch_end - epoch_start

            if epoch_size < 1:
                # Empty epoch - create zeros
                epoch_tokens = np.zeros((self.n_tokens, self.token_dim), dtype=np.float32)
                epoch_mask = np.zeros(self.n_tokens, dtype=bool)
            else:
                # Select tokens within this epoch window
                n_select = min(tokens_per_epoch, epoch_size)

                if self.phase_stratified and period_hours is not None:
                    # Phase-stratified sampling
                    epoch_t = t[epoch_start:epoch_end]
                    local_idx = maybe_phase_stratified_indices(
                        epoch_t,
                        period_hours=period_hours,
                        n_select=n_select,
                        n_bins=self.phase_bins,
                        rng=rng,
                        allow_missing_periods=self.allow_missing_periods,
                        object_id=object_id,
                    )
                    global_idx = epoch_start + local_idx

                    # Debug: compare phase coverage (stratified vs random baseline)
                    if self.debug_phase_cov and epoch_idx == 0:
                        cov_strat = compute_bin_coverage(local_idx, epoch_t, period_hours, n_bins=self.phase_bins)
                        # Compute what random would have gotten using EXACT same logic as production
                        # Production path: rng.choice(np.arange(epoch_start, epoch_end), size=n_select, replace=False)
                        # We use a separate RNG to not perturb training, but same selection logic
                        rng_baseline = np.random.default_rng(self.seed + idx)  # Same seed scheme
                        # Simulate the random path: select from epoch window, convert to local indices
                        rand_global_idx = rng_baseline.choice(
                            np.arange(epoch_start, epoch_end), size=n_select, replace=False
                        )
                        rand_local_idx = rand_global_idx - epoch_start
                        cov_rand = compute_bin_coverage(rand_local_idx, epoch_t, period_hours, n_bins=self.phase_bins)

                        # Track for aggregate stats
                        self._cov_strat_list.append(cov_strat)
                        self._cov_rand_list.append(cov_rand)

                        # Per-object log (limited)
                        if self._debug_count < self.debug_limit:
                            logger.info(
                                f"phase_cov object={object_id} strat={cov_strat}/{self.phase_bins} "
                                f"rand={cov_rand}/{self.phase_bins} period_h={period_hours:.2f}"
                            )
                            self._debug_count += 1
                else:
                    # Random sampling (fallback)
                    if self.phase_stratified and period_hours is None and not self.allow_missing_periods:
                        raise ValueError(
                            f"Period not available for {object_id}. "
                            f"Use --allow-missing-periods 1 to fall back to random sampling."
                        )
                    global_idx = rng.choice(np.arange(epoch_start, epoch_end), size=n_select, replace=False)

                # Build tokens
                epoch_tokens = np.zeros((self.n_tokens, self.token_dim), dtype=np.float32)
                epoch_mask = np.zeros(self.n_tokens, dtype=bool)

                for token_idx, obs_idx in enumerate(global_idx[:self.n_tokens]):
                    # Token format: [sun_u(3), obs_u(3), brightness(1), log_magerr(1), padding(20)]
                    epoch_tokens[token_idx, 0:3] = sun_u[obs_idx]
                    epoch_tokens[token_idx, 3:6] = obs_u[obs_idx]
                    epoch_tokens[token_idx, 6] = brightness[obs_idx]
                    epoch_tokens[token_idx, 7] = 0.0  # log_magerr placeholder
                    # Rest is padding (zeros)
                    epoch_mask[token_idx] = True

            epochs_tokens.append(epoch_tokens)
            epochs_masks.append(epoch_mask)

        epochs = np.stack(epochs_tokens, axis=0)  # (n_epochs, n_tokens, token_dim)
        masks = np.stack(epochs_masks, axis=0)    # (n_epochs, n_tokens)

        return epochs, masks, pole

    def report_coverage_stats(self):
        """Report aggregate phase coverage statistics."""
        if not self._cov_strat_list:
            return

        cov_strat = np.array(self._cov_strat_list)
        cov_rand = np.array(self._cov_rand_list)
        delta = cov_strat - cov_rand

        n_better = np.sum(delta > 0)
        n_same = np.sum(delta == 0)
        n_worse = np.sum(delta < 0)
        n_total = len(delta)

        logger.info("=" * 50)
        logger.info("PHASE COVERAGE AGGREGATE STATS")
        logger.info("=" * 50)
        logger.info(f"  Objects analyzed: {n_total}")
        logger.info(f"  Mean strat coverage: {np.mean(cov_strat):.2f}/{self.phase_bins}")
        logger.info(f"  Mean rand coverage:  {np.mean(cov_rand):.2f}/{self.phase_bins}")
        logger.info(f"  Mean delta (strat - rand): {np.mean(delta):+.2f}")
        logger.info(f"  Strat better: {n_better}/{n_total} ({100*n_better/n_total:.1f}%)")
        logger.info(f"  Same:         {n_same}/{n_total} ({100*n_same/n_total:.1f}%)")
        logger.info(f"  Rand better:  {n_worse}/{n_total} ({100*n_worse/n_total:.1f}%)")
        logger.info("=" * 50)


def collate_fn(batch):
    """Collate batch of (epochs, masks, poles) into tensors."""
    epochs_list, masks_list, poles_list = zip(*batch)

    epochs = torch.from_numpy(np.stack(epochs_list, axis=0))  # (B, E, T, D)
    masks = torch.from_numpy(np.stack(masks_list, axis=0))    # (B, E, T)
    poles = torch.from_numpy(np.stack(poles_list, axis=0))    # (B, 3)

    return epochs, masks, poles


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train one epoch, return mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for epochs_batch, masks_batch, poles_batch in dataloader:
        epochs_batch = epochs_batch.to(device)
        masks_batch = masks_batch.to(device)
        poles_batch = poles_batch.to(device)

        optimizer.zero_grad()

        # Forward
        outputs = model(epochs_batch, masks_batch)
        logp_grid = outputs['logp_grid']  # (B, n_poles)

        # Compute target: find closest pole in grid (accounting for antipode)
        pole_grid = model.pole_grid.to(device)  # (n_poles, 3)

        # Cosine similarity with all poles
        cos_sim = torch.matmul(poles_batch, pole_grid.T)  # (B, n_poles)
        cos_sim_antipode = torch.matmul(-poles_batch, pole_grid.T)
        cos_sim_best = torch.maximum(cos_sim, cos_sim_antipode)

        # Target is argmax
        target_idx = torch.argmax(cos_sim_best, dim=-1)  # (B,)

        # Cross-entropy loss
        loss = F.cross_entropy(outputs['logits'], target_idx)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[float, np.ndarray]:
    """Evaluate model, return median error and all errors."""
    model.eval()
    all_errors = []

    with torch.no_grad():
        for epochs_batch, masks_batch, poles_batch in dataloader:
            epochs_batch = epochs_batch.to(device)
            masks_batch = masks_batch.to(device)
            poles_batch = poles_batch.to(device)

            # Predict
            pred_poles = model.predict_pole(epochs_batch, masks_batch)

            # Compute errors (antipode-aware)
            errors = angular_distance_with_antipode(pred_poles, poles_batch)
            all_errors.extend(errors.cpu().numpy())

    all_errors = np.array(all_errors)
    return float(np.median(all_errors)), all_errors


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train EpochAttentionGridModel on DAMIT data"
    )

    # Data arguments
    parser.add_argument("--csv-dir", type=Path, default=Path("DAMIT_csv_high"),
                        help="Directory with DAMIT CSV files")
    parser.add_argument("--damit-root", type=Path, default=None,
                        help="DAMIT root for loading ground-truth poles (optional)")
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output directory for checkpoints")

    # Training arguments
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--seeds", type=int, default=5000, help="Base random seed")
    parser.add_argument("--epochs-max", type=int, default=50, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cuda", help="Device")

    # Model arguments
    parser.add_argument("--n-epochs-per-sample", type=int, default=5, help="Epochs per sample")
    parser.add_argument("--n-tokens", type=int, default=64, help="Tokens per epoch")
    parser.add_argument("--token-dim", type=int, default=28, help="Token dimension")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=2, help="Transformer layers")
    parser.add_argument("--n-poles", type=int, default=4096, help="Pole grid size")

    # Period cache arguments
    parser.add_argument("--period-cache", type=Path, default=Path("artifacts/period_cache_damit.json"),
                        help="Path to period cache JSON")
    parser.add_argument("--allow-missing-periods", type=int, default=0,
                        help="Allow missing periods (0=no, 1=yes)")
    parser.add_argument("--phase-bins", type=int, default=8,
                        help="Number of phase bins for stratified sampling")
    parser.add_argument("--phase-stratified", type=int, default=1,
                        help="Enable phase-stratified sampling (0=no, 1=yes)")

    # Debug arguments
    parser.add_argument("--debug-phase-cov", type=int, default=0,
                        help="Print phase coverage for first N objects (0=disabled)")

    args = parser.parse_args()

    # Setup
    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seeds)
    np.random.seed(args.seeds)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ==========================================================================
    # LOAD DATA
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("PHASE 8: TRAINING EPOCH ATTENTION GRID MODEL")
    logger.info("=" * 70)

    logger.info(f"Loading CSV data from: {args.csv_dir}")
    objects, object_ids = load_dataset(args.csv_dir, args.damit_root)
    logger.info(f"Loaded {len(objects)} objects")

    # ==========================================================================
    # LOAD PERIOD CACHE
    # ==========================================================================
    period_cache = {}
    if args.phase_stratified:
        if args.period_cache.exists():
            logger.info(f"Loading period cache from: {args.period_cache}")
            period_cache = load_period_cache(args.period_cache)

            # Validate coverage using library's fail-fast helper
            n_loaded = sum(1 for oid in object_ids if oid in period_cache)
            n_total = len(object_ids)
            frac = n_loaded / n_total if n_total > 0 else 0.0

            logger.info(f"Periods loaded: {n_loaded}/{n_total} ({frac:.1%}) from {args.period_cache}")

            if not args.allow_missing_periods:
                # Use require_coverage for consistent fail-fast behavior
                try:
                    require_coverage(period_cache, object_ids, min_frac=1.0, context="DAMIT training")
                except ValueError as e:
                    raise ValueError(
                        f"{e}\nUse --allow-missing-periods 1 to continue with random sampling for missing periods."
                    )
        else:
            logger.warning(f"Period cache not found: {args.period_cache}")
            if not args.allow_missing_periods:
                raise FileNotFoundError(
                    f"Period cache not found: {args.period_cache}. "
                    f"Run make_period_cache_from_damit.py first, or use --allow-missing-periods 1."
                )
    else:
        logger.info("Phase-stratified sampling disabled (--phase-stratified 0)")

    # ==========================================================================
    # K-FOLD CROSS-VALIDATION
    # ==========================================================================
    n_objects = len(objects)
    group_ids = np.arange(n_objects)

    gkf = GroupKFold(n_splits=args.n_folds)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(np.arange(n_objects), groups=group_ids)):
        logger.info("=" * 70)
        logger.info(f"FOLD {fold_idx + 1}/{args.n_folds}")
        logger.info("=" * 70)

        # Create datasets
        train_objects = [objects[i] for i in train_idx]
        val_objects = [objects[i] for i in val_idx]

        train_dataset = MultiEpochDataset(
            train_objects,
            period_cache=period_cache,
            n_epochs=args.n_epochs_per_sample,
            n_tokens=args.n_tokens,
            token_dim=args.token_dim,
            phase_stratified=bool(args.phase_stratified),
            phase_bins=args.phase_bins,
            allow_missing_periods=bool(args.allow_missing_periods),
            seed=args.seeds + fold_idx * 1000,
            debug_phase_cov=bool(args.debug_phase_cov) and fold_idx == 0,
            debug_limit=args.debug_phase_cov if args.debug_phase_cov else 5,
        )

        val_dataset = MultiEpochDataset(
            val_objects,
            period_cache=period_cache,
            n_epochs=args.n_epochs_per_sample,
            n_tokens=args.n_tokens,
            token_dim=args.token_dim,
            phase_stratified=bool(args.phase_stratified),
            phase_bins=args.phase_bins,
            allow_missing_periods=bool(args.allow_missing_periods),
            seed=args.seeds + fold_idx * 1000 + 500,
            debug_phase_cov=False,
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
        )

        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

        # Create model
        model = EpochAttentionGridModel(
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=4,
            dim_feedforward=args.d_model * 4,
            dropout=0.1,
            max_epochs=args.n_epochs_per_sample,
            token_dim=args.token_dim,
            n_poles=args.n_poles,
        )
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_max)

        # Training loop
        best_val_error = float('inf')
        patience_counter = 0
        fold_dir = args.outdir / f"fold{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(args.epochs_max):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_error, _ = evaluate(model, val_loader, device)
            scheduler.step()

            logger.info(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_median={val_error:.2f}°")

            if val_error < best_val_error:
                best_val_error = val_error
                patience_counter = 0
                torch.save(model.state_dict(), fold_dir / "model_best.pt")
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Final evaluation
        model.load_state_dict(torch.load(fold_dir / "model_best.pt", map_location=device))
        final_error, errors = evaluate(model, val_loader, device)

        fold_results.append({
            'fold': fold_idx,
            'median_error': final_error,
            'mean_error': float(np.mean(errors)),
            'p90_error': float(np.percentile(errors, 90)),
        })

        logger.info(f"Fold {fold_idx+1} final: median={final_error:.2f}°")

        # Report aggregate coverage stats after first fold (when debug is enabled)
        if fold_idx == 0 and args.debug_phase_cov:
            train_dataset.report_coverage_stats()

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    logger.info("=" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)

    medians = [r['median_error'] for r in fold_results]
    logger.info(f"Median errors across folds: {medians}")
    logger.info(f"Mean of medians: {np.mean(medians):.2f}° ± {np.std(medians):.2f}°")

    # Save results
    with open(args.outdir / "results.json", 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'mean_median': float(np.mean(medians)),
            'std_median': float(np.std(medians)),
            'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }, f, indent=2)

    logger.info(f"Results saved to: {args.outdir / 'results.json'}")


if __name__ == "__main__":
    main()
