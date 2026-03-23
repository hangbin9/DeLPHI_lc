"""Training script for GeoHierK3Transformer with 4-term production loss.

Loss: oracle_softmin + continuous_diversity(sigma=15 deg) + batch_variance(lambda=5)
      + similarity_matching(lambda=2)

174 DAMIT asteroids (QF>=3) is the production dataset, with 5-fold CV.
For other datasets, update --csv-dir and corresponding data paths.

Usage (Multi-Epoch, Recommended):
    python -m lc_pipeline.scripts.train_k3 \\
        --csv-dir data/damit_csv_qf_ge_3 \\
        --spin-root data/damit_spins_complete \\
        --period-json data/periods.json \\
        --outdir lc_pipeline/checkpoints \\
        --folds 0,1,2,3,4 \\
        --device cuda \\
        --seed 777 \\
        --epochs 200 \\
        --patience 50 \\
        --dataset-mode single_epoch

Usage (Legacy Aggregated):
    python -m lc_pipeline.scripts.train_k3 \\
        --csv-dir data/damit_csv_qf_ge_3 \\
        --spin-root data/damit_spins_complete \\
        --outdir lc_pipeline/checkpoints \\
        --folds 0 \\
        --dataset-mode aggregated \\
        --epochs 200 \\
        --patience 50

Period JSON format (--period-json):
    {"asteroid_101": 8.34, "asteroid_102": 5.22, ...}
    Maps asteroid ID to rotation period in hours. If omitted, period estimation
    is performed automatically during training (slower but does not require
    pre-computed periods).

Dataset Modes:
- single_epoch (default): Each epoch is a separate sample (2,987 samples)
  - 17.2x more training data
  - Faster convergence (~6 epochs median with early stopping)
  - Less overfitting
  - Production performance: 19.02 deg mean oracle (asteroid-level)

- aggregated (legacy): All epochs per asteroid aggregated (174 samples)
  - Original baseline mode
  - Slower convergence, more epochs needed
  - Higher overfitting risk
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lc_pipeline.data.damit_multiepoch_dataset import create_dataloaders
from lc_pipeline.data.single_epoch_dataset import create_single_epoch_dataloaders
from lc_pipeline.models.geo_hier_k3_transformer import GeoHierK3Transformer
from lc_pipeline.training.losses_axisnet import combined_loss
from lc_pipeline.evaluation.eval_axisnet import evaluate_fold

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_object_ids(csv_dir: Path) -> List[str]:
    """Load list of object IDs from CSV directory."""
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob("*.csv"))
    object_ids = [f.stem for f in csv_files]
    return object_ids


def deterministic_fold_split(object_ids: List[str], n_folds: int = 3, seed: int = 1337) -> Tuple[List[List[str]], List[List[str]]]:
    """Deterministically split objects into train/val folds."""
    np.random.seed(seed)
    shuffled = np.random.permutation(object_ids)

    fold_size = len(shuffled) // n_folds
    folds_train = []
    folds_val = []

    for fold_idx in range(n_folds):
        val_indices = np.arange(fold_idx * fold_size, (fold_idx + 1) * fold_size)
        val_ids = [shuffled[i] for i in val_indices]
        train_ids = [shuffled[i] for i in range(len(shuffled)) if i not in val_indices]

        folds_train.append(train_ids)
        folds_val.append(val_ids)

    return folds_train, folds_val


class TrainerK3:
    """Trainer for 29O-R K=3 model."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        max_epochs: int = 200,
        softmin_tau_deg: float = 5.0,
        lambda_div: float = 0.5,
        div_sigma_deg: float = 15.0,
        lambda_var: float = 5.0,
        lambda_sim: float = 2.0,
        lambda_q: float = 0.0,
        gap_tau_deg: float = 10.0,
        ignore_near_ties_deg: float = 1.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )

        # Loss hyperparameters (matching production checkpoint config)
        self.softmin_tau_deg = softmin_tau_deg
        self.lambda_div = lambda_div
        self.div_sigma_deg = div_sigma_deg
        self.lambda_var = lambda_var
        self.lambda_sim = lambda_sim
        self.lambda_q = lambda_q
        self.gap_tau_deg = gap_tau_deg
        self.ignore_near_ties_deg = ignore_near_ties_deg

        self.best_loss = float('inf')
        self.patience = 0

    def train_epoch(self, train_loader: DataLoader, epoch: Optional[int] = None, total_epochs: int = 200) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            tokens = batch['tokens'].to(self.device)  # [B, W, T, 9]
            mask = batch['mask'].to(self.device)      # [B, W, T]
            solutions_list = batch['solutions']        # List of [S, 3] or None

            # Convert solutions to GPU if present
            solutions_list = [
                s.to(self.device) if s is not None else None
                for s in solutions_list
            ]

            self.optimizer.zero_grad()

            poles, quality_logits = self.model(tokens, mask)  # [B, 3, 3], [B, 3] or None

            loss_dict = combined_loss(
                poles, quality_logits, solutions_list,
                lambda_div=self.lambda_div,
                lambda_q=self.lambda_q,
                lambda_var=self.lambda_var,
                lambda_sim=self.lambda_sim,
                softmin_tau_deg=self.softmin_tau_deg,
                div_sigma_deg=self.div_sigma_deg,
                gap_tau_deg=self.gap_tau_deg,
                ignore_near_ties_deg=self.ignore_near_ties_deg,
            )
            loss = loss_dict['loss']

            loss_val = loss.item()

            # Skip batch if loss is NaN or Inf
            if np.isnan(loss_val) or np.isinf(loss_val):
                self.optimizer.zero_grad()
                n_batches += 1
                continue

            loss.backward()

            # Check for NaN/Inf gradients and skip if found
            has_invalid_grad = False
            for param in self.model.parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        has_invalid_grad = True
                        break

            if has_invalid_grad:
                logger.warning(f"Skipping batch {n_batches} due to invalid gradients")
                self.optimizer.zero_grad()
                n_batches += 1
                continue

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Final check: ensure no NaN in model parameters after clipping
            has_nan_param = False
            for param in self.model.parameters():
                if torch.isnan(param).any() or torch.isinf(param).any():
                    has_nan_param = True
                    break

            if has_nan_param:
                logger.warning(f"Skipping batch {n_batches} - model has NaN parameters")
                self.optimizer.zero_grad()
                n_batches += 1
                continue

            self.optimizer.step()

            epoch_loss += loss_val
            n_batches += 1

        return epoch_loss / max(n_batches, 1)

    def validate(self, val_loader: DataLoader) -> Tuple[float, dict]:
        """Validate on fold."""
        self.model.eval()
        fold_results = evaluate_fold(self.model, val_loader, device=self.device)

        metrics = fold_results['metrics']
        oracle_median = metrics['oracle_median_deg']

        if oracle_median is None:
            return float('inf'), metrics

        return float(oracle_median), metrics

    def should_stop(self, val_loss: float, patience_limit: int = 25) -> bool:
        """Check early stopping criterion."""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience = 0
            return False
        else:
            self.patience += 1
            return self.patience >= patience_limit

    def save_checkpoint(self, checkpoint_path: Path, config: dict, fold_idx: int, epoch: int):
        """Save model checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': config,
            'fold_idx': fold_idx,
            'epoch': epoch,
            'best_loss': self.best_loss,
        }, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")


def train_fold(
    fold_idx: int,
    train_ids: List[str],
    val_ids: List[str],
    csv_dir: Path,
    spin_root: Optional[Path],
    outdir: Path,
    config: dict,
) -> dict:
    """Train one fold."""
    dataset_mode = config.get('dataset_mode', 'single_epoch')
    min_gap_days = config.get('min_gap_days', 30.0)

    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_idx}: train={len(train_ids)}, val={len(val_ids)}")
    logger.info(f"Dataset mode: {dataset_mode}")
    logger.info(f"{'='*60}\n")

    # Create dataloaders based on dataset mode
    if dataset_mode == 'single_epoch':
        # Multi-Epoch: Each epoch is a separate sample (recommended)
        logger.info(f"Using single_epoch dataset with min_gap_days={min_gap_days}")
        train_loader, val_loader = create_single_epoch_dataloaders(
            csv_dir=csv_dir,
            spin_root=spin_root,
            train_ids=train_ids,
            val_ids=val_ids,
            batch_size=config['batch_size'],
            min_gap_days=min_gap_days,
            use_geometry=config.get('use_geometry', False),
            period_json=config.get('period_json'),
        )
    else:
        # Aggregated: All epochs per asteroid (legacy mode)
        logger.info("Using aggregated dataset (legacy mode)")
        train_loader, val_loader = create_dataloaders(
            csv_dir=csv_dir,
            object_ids=train_ids + val_ids,
            train_ids=train_ids,
            val_ids=val_ids,
            spin_root=spin_root,
            batch_size=config['batch_size'],
            seed=config['seed'],
            augment=config.get('enable_augmentation', False),
        )

    # Create K=3 model
    device = config['device']
    model = GeoHierK3Transformer(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers_window=config['n_layers_window'],
        n_layers_cross=config['n_layers_cross'],
        n_feature_input=13,  # V14 baseline: 3 temporal + 6 geometry (zeros) + 4 period
        include_quality_head=config['include_quality_head'],
        dropout=config['dropout'],
    )

    # Create trainer
    trainer = TrainerK3(
        model=model,
        device=device,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        max_epochs=config['epochs'],
        softmin_tau_deg=config['softmin_tau_deg'],
        lambda_div=config['lambda_div'],
        div_sigma_deg=config['div_sigma_deg'],
        lambda_var=config['lambda_var'],
        lambda_sim=config['lambda_sim'],
        lambda_q=config['lambda_q'],
        gap_tau_deg=config['gap_tau_deg'],
        ignore_near_ties_deg=config['ignore_near_ties_deg'],
    )

    # Training loop
    fold_outdir = Path(outdir) / f"fold_{fold_idx}"
    fold_outdir.mkdir(parents=True, exist_ok=True)

    train_history = []
    val_history = []

    # Track both oracle and quality for checkpoint selection
    best_oracle_error = float('inf')
    best_oracle_epoch = -1
    best_quality_error = float('inf')
    best_quality_epoch = -1

    for epoch in range(config['epochs']):
        train_loss = trainer.train_epoch(train_loader, epoch=epoch, total_epochs=config['epochs'])
        val_loss, val_metrics = trainer.validate(val_loader)

        train_history.append(train_loss)
        val_history.append({
            'epoch': epoch,
            'loss': val_loss,
            'metrics': val_metrics,
        })

        # Log progress
        oracle_med = val_metrics['oracle_median_deg'] if val_metrics['oracle_median_deg'] is not None else float('nan')
        quality_med = val_metrics.get('quality_median_deg')

        log_msg = (
            f"Fold {fold_idx} | Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} | Val Oracle: {oracle_med:.2f}°"
        )
        if quality_med is not None:
            selector_acc = val_metrics.get('selector_accuracy', 0.0)
            log_msg += f" | Val Quality: {quality_med:.2f}° | Sel.Acc: {selector_acc:.3f}"
        logger.info(log_msg)

        # LR scheduler step
        trainer.scheduler.step()

        # Track and save best ORACLE model (for early stopping alignment)
        if oracle_med < best_oracle_error:
            best_oracle_error = oracle_med
            best_oracle_epoch = epoch
            logger.info(f"  → New best oracle: {oracle_med:.2f}° (saving checkpoint)")
            trainer.save_checkpoint(
                fold_outdir / "checkpoint_best_oracle.pt",
                config=config,
                fold_idx=fold_idx,
                epoch=epoch,
            )

        # Track and save best QUALITY model (only when quality head is enabled)
        if quality_med is not None and quality_med < best_quality_error:
            best_quality_error = quality_med
            best_quality_epoch = epoch
            logger.info(f"  → New best quality: {quality_med:.2f}° (saving checkpoint)")
            trainer.save_checkpoint(
                fold_outdir / "checkpoint_best_quality.pt",
                config=config,
                fold_idx=fold_idx,
                epoch=epoch,
            )

        # Early stopping based on oracle (aligned with oracle checkpoint)
        if trainer.should_stop(val_loss, patience_limit=config['patience']):
            logger.info(f"Early stopping at epoch {epoch}")
            logger.info(f"Best oracle was {best_oracle_error:.2f}° at epoch {best_oracle_epoch}")
            logger.info(f"Best quality was {best_quality_error:.2f}° at epoch {best_quality_epoch}")
            break

        # Save periodic checkpoints
        if epoch % 50 == 0:
            trainer.save_checkpoint(
                fold_outdir / f"checkpoint_epoch_{epoch}.pt",
                config=config,
                fold_idx=fold_idx,
                epoch=epoch,
            )

    # Save final model (CRITICAL: ensures we always have the final trained model)
    logger.info(f"Fold {fold_idx}: Saving final model...")
    trainer.save_checkpoint(
        fold_outdir / "checkpoint_final.pt",
        config=config,
        fold_idx=fold_idx,
        epoch=epoch,  # epoch from the training loop
    )

    # Save training history
    history_file = fold_outdir / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump({
            'train_loss': train_history,
            'val_history': val_history,
        }, f, indent=2)

    # Final evaluation
    logger.info(f"\nFold {fold_idx} final evaluation...")
    final_metrics = trainer.validate(val_loader)[1]

    return {
        'fold_idx': fold_idx,
        'n_train': len(train_ids),
        'n_val': len(val_ids),
        'final_metrics': final_metrics,
        'history_file': str(history_file),
    }


def main():
    parser = argparse.ArgumentParser(description='Train AxisNet 29O-R K=3 model')

    # Data
    parser.add_argument('--csv-dir', type=Path, required=True, help='Path to DAMIT CSV directory')
    parser.add_argument('--spin-root', type=Path, default=None, help='Path to spin solutions directory')
    parser.add_argument('--outdir', type=Path, required=True, help='Output directory for checkpoints')

    # Folds
    parser.add_argument('--folds', type=str, default='0', help='Comma-separated fold indices')
    parser.add_argument('--n-folds', type=int, default=5, help='Total number of folds')
    parser.add_argument('--split-dir', type=Path, default=None,
                        help='Directory with pre-computed fold JSONs (fold{i}_train.json / fold{i}_test.json). '
                             'When provided, overrides deterministic_fold_split().')

    # Training
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=777, help='Random seed')
    parser.add_argument('--epochs', type=int, default=200, help='Max epochs')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Model architecture
    parser.add_argument('--d-model', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--n-layers-window', type=int, default=4, help='Transformer layers in window encoder')
    parser.add_argument('--n-layers-cross', type=int, default=0, help='Transformer layers in cross-window encoder (0 = disabled)')

    # Loss hyperparameters (defaults match production checkpoints)
    parser.add_argument('--softmin-tau-deg', type=float, default=5.0, help='Softmin temperature (degrees)')
    parser.add_argument('--lambda-div', type=float, default=0.5, help='Continuous diversity loss weight')
    parser.add_argument('--div-sigma-deg', type=float, default=15.0, help='Diversity sigma (degrees)')
    parser.add_argument('--lambda-var', type=float, default=5.0, help='Batch variance loss weight')
    parser.add_argument('--lambda-sim', type=float, default=2.0, help='Similarity-matching loss weight')
    parser.add_argument('--lambda-q', type=float, default=0.0, help='Quality loss weight (0 = disabled)')
    parser.add_argument('--gap-tau-deg', type=float, default=10.0, help='Gap tau for quality weighting (degrees)')
    parser.add_argument('--ignore-near-ties-deg', type=float, default=1.0, help='Ignore gap below this threshold (degrees)')

    parser.add_argument('--enable-augmentation', action='store_true', help='Enable data augmentation during training')

    # Dataset mode (multi-epoch vs aggregated)
    parser.add_argument('--dataset-mode', type=str, default='single_epoch',
                        choices=['single_epoch', 'aggregated'],
                        help='Dataset mode: single_epoch (recommended) or aggregated (legacy)')
    parser.add_argument('--min-gap-days', type=float, default=30.0,
                        help='Minimum gap (days) for epoch detection (single_epoch mode only)')
    parser.add_argument('--use-geometry', action='store_true',
                        help='Use active geometry features (default: zeros for baseline)')
    parser.add_argument('--period-json', type=str, required=True,
                        help='Path to JSON with DAMIT periods per asteroid (REQUIRED). '
                             'Format: {"asteroid_101": {"period_hours": 7.81}, ...}')

    args = parser.parse_args()

    # Set random seeds (including CUDA for reproducibility)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load object IDs and create fold splits
    logger.info(f"Loading objects from {args.csv_dir}")
    object_ids = load_object_ids(args.csv_dir)
    logger.info(f"Found {len(object_ids)} objects")

    if args.split_dir is not None:
        # Use pre-computed JSON fold splits
        split_dir = Path(args.split_dir)
        logger.info(f"Loading fold splits from {split_dir}")
        folds_train = []
        folds_val = []
        for i in range(args.n_folds):
            train_file = split_dir / f"fold{i}_train.json"
            test_file = split_dir / f"fold{i}_test.json"
            with open(train_file) as f:
                folds_train.append(json.load(f))
            with open(test_file) as f:
                folds_val.append(json.load(f))
            logger.info(f"  Fold {i}: train={len(folds_train[-1])}, val={len(folds_val[-1])}")
    else:
        folds_train, folds_val = deterministic_fold_split(object_ids, n_folds=args.n_folds, seed=args.seed)

    # Parse fold indices
    fold_indices = [int(f) for f in args.folds.split(',')]

    # Config dict for checkpointing
    config = {
        'csv_dir': str(args.csv_dir),
        'spin_root': str(args.spin_root) if args.spin_root else None,
        'device': args.device,
        'seed': args.seed,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'dropout': args.dropout,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers_window': args.n_layers_window,
        'n_layers_cross': args.n_layers_cross,
        'n_feature_input': 13,  # V14 baseline: 3 temporal + 6 geometry (zeros) + 4 period
        'include_quality_head': args.lambda_q > 0,
        'softmin_tau_deg': args.softmin_tau_deg,
        'lambda_div': args.lambda_div,
        'div_sigma_deg': args.div_sigma_deg,
        'lambda_var': args.lambda_var,
        'lambda_sim': args.lambda_sim,
        'lambda_q': args.lambda_q,
        'gap_tau_deg': args.gap_tau_deg,
        'ignore_near_ties_deg': args.ignore_near_ties_deg,
        'enable_augmentation': args.enable_augmentation,
        # Multi-epoch settings
        'dataset_mode': args.dataset_mode,
        'min_gap_days': args.min_gap_days,
        'use_geometry': args.use_geometry,
        'period_json': args.period_json,
    }

    # Save config
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_file = outdir / 'config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved config to {config_file}")

    # Train folds
    fold_results = []
    for fold_idx in fold_indices:
        result = train_fold(
            fold_idx=fold_idx,
            train_ids=folds_train[fold_idx],
            val_ids=folds_val[fold_idx],
            csv_dir=args.csv_dir,
            spin_root=args.spin_root,
            outdir=outdir,
            config=config,
        )
        fold_results.append(result)

    # Save fold results summary
    results_file = outdir / 'fold_results.json'
    with open(results_file, 'w') as f:
        json.dump(fold_results, f, indent=2)
    logger.info(f"Saved fold results to {results_file}")

    logger.info(f"\n{'='*60}")
    logger.info("Training complete! (K=3 variant)")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
