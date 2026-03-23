#!/usr/bin/env python3
"""
train_pole_model.py - Train a DeLPHI pole prediction model on asteroid lightcurve data.

This script trains the GeoHierK3Transformer model using multi-epoch training on
DAMIT lightcurve data. It uses the same training logic that produced the shipped
checkpoints.

Requirements:
    - DAMIT lightcurve CSV files (8 columns: time, brightness, sun_xyz, obs_xyz)
    - Ground truth spin solutions (JSON files with pole vectors)
    - Period JSON mapping asteroid IDs to rotation periods in hours

Usage Examples:
    # Train all 5 folds (production recipe)
    python train_pole_model.py \\
        --csv-dir data/damit_csv_qf_ge_3 \\
        --spin-root data/damit_spins_complete \\
        --period-json data/periods.json \\
        --outdir checkpoints_new \\
        --folds 0,1,2,3,4

    # Train a single fold for testing
    python train_pole_model.py \\
        --csv-dir data/damit_csv_qf_ge_3 \\
        --spin-root data/damit_spins_complete \\
        --period-json data/periods.json \\
        --outdir checkpoints_new \\
        --folds 0

    # Quick test run (2 epochs, CPU)
    python train_pole_model.py \\
        --csv-dir data/damit_csv_qf_ge_3 \\
        --spin-root data/damit_spins_complete \\
        --period-json data/periods.json \\
        --outdir test_output \\
        --folds 0 --epochs 2 --device cpu

Data Setup:
    See docs/USER_GUIDE.md for instructions on downloading DAMIT data and
    preparing the required directory structure.

    Expected directory layout:
        data/
        ├── damit_csv_qf_ge_3/          # Lightcurve CSV files
        │   ├── asteroid_101.csv
        │   └── ...
        ├── damit_spins_complete/        # Ground truth pole solutions
        │   ├── asteroid_101.json
        │   └── ...
        └── periods.json                 # {"asteroid_101": 8.34, ...}

Output:
    - Checkpoint files (.pt) in --outdir/fold_N/
    - Training history (JSON) per fold
    - Config JSON saved to --outdir/config.json

Training Details:
    - Model: GeoHierK3Transformer (d_model=128, 4 heads, 4 layers, ~994K params)
    - Loss: oracle_softmin + continuous_diversity + batch_variance + contrastive
    - Dataset: Multi-epoch (each observing epoch is a separate training sample)
    - Optimizer: AdamW (lr=3e-4, weight_decay=1e-4)
    - Scheduler: CosineAnnealingLR
    - Early stopping: patience=50 on validation oracle error
    - Typical training time: ~5 minutes per fold on GPU
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.optim as optim
except ImportError:
    print("ERROR: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

from lc_pipeline.data.single_epoch_dataset import create_single_epoch_dataloaders
from lc_pipeline.data.damit_multiepoch_dataset import create_dataloaders
from lc_pipeline.models.geo_hier_k3_transformer import GeoHierK3Transformer
from lc_pipeline.training.losses_axisnet import combined_loss
from lc_pipeline.evaluation.eval_axisnet import evaluate_fold

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_object_ids(csv_dir: Path):
    """Load list of object IDs from CSV directory."""
    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    return [f.stem for f in csv_files]


def deterministic_fold_split(object_ids, n_folds=5, seed=1337):
    """Deterministically split objects into train/val folds."""
    np.random.seed(seed)
    shuffled = np.random.permutation(object_ids)
    fold_size = len(shuffled) // n_folds

    folds_train, folds_val = [], []
    for fold_idx in range(n_folds):
        val_indices = set(range(fold_idx * fold_size, (fold_idx + 1) * fold_size))
        val_ids = [shuffled[i] for i in val_indices]
        train_ids = [shuffled[i] for i in range(len(shuffled)) if i not in val_indices]
        folds_train.append(train_ids)
        folds_val.append(val_ids)

    return folds_train, folds_val


def train_fold(fold_idx, train_ids, val_ids, csv_dir, spin_root, outdir, config):
    """Train one fold of the model."""
    logger.info(f"\n{'='*60}")
    logger.info(f"FOLD {fold_idx}: train={len(train_ids)} asteroids, val={len(val_ids)} asteroids")
    logger.info(f"{'='*60}\n")

    device = config['device']

    # Create dataloaders (multi-epoch mode by default)
    train_loader, val_loader = create_single_epoch_dataloaders(
        csv_dir=csv_dir,
        spin_root=spin_root,
        train_ids=train_ids,
        val_ids=val_ids,
        batch_size=config['batch_size'],
        min_gap_days=config.get('min_gap_days', 30.0),
        use_geometry=False,
        period_json=config.get('period_json'),
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    model = GeoHierK3Transformer(
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers_window=config['n_layers_window'],
        n_layers_cross=config['n_layers_cross'],
        n_feature_input=13,
        include_quality_head=False,
        dropout=config['dropout'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])

    # Training loop
    fold_outdir = Path(outdir) / f"fold_{fold_idx}"
    fold_outdir.mkdir(parents=True, exist_ok=True)

    best_oracle = float('inf')
    best_epoch = -1
    patience_counter = 0
    train_history = []

    for epoch in range(config['epochs']):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            tokens = batch['tokens'].to(device)
            mask = batch['mask'].to(device)
            solutions_list = [
                s.to(device) if s is not None else None
                for s in batch['solutions']
            ]

            optimizer.zero_grad()
            poles, quality_logits = model(tokens, mask)

            loss_dict = combined_loss(
                poles, quality_logits, solutions_list,
                lambda_div=config['lambda_div'],
                lambda_q=0.0,
                lambda_var=config['lambda_var'],
                lambda_sim=config['lambda_sim'],
                softmin_tau_deg=config['softmin_tau_deg'],
                div_sigma_deg=config['div_sigma_deg'],
                gap_tau_deg=config['gap_tau_deg'],
                ignore_near_ties_deg=config['ignore_near_ties_deg'],
            )
            loss = loss_dict['loss']
            loss_val = loss.item()

            if np.isnan(loss_val) or np.isinf(loss_val):
                optimizer.zero_grad()
                n_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss_val
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step()

        # --- Validate ---
        model.eval()
        fold_results = evaluate_fold(model, val_loader, device=device)
        oracle_med = fold_results['metrics']['oracle_median_deg']
        if oracle_med is None:
            oracle_med = float('inf')

        train_history.append({
            'epoch': epoch,
            'train_loss': avg_loss,
            'oracle_median_deg': float(oracle_med),
        })

        logger.info(
            f"Fold {fold_idx} | Epoch {epoch:3d} | "
            f"Train Loss: {avg_loss:.4f} | Val Oracle Median: {oracle_med:.2f} deg"
        )

        # Best model tracking
        if oracle_med < best_oracle:
            best_oracle = oracle_med
            best_epoch = epoch
            patience_counter = 0
            logger.info(f"  -> New best: {oracle_med:.2f} deg (saving checkpoint)")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'fold_idx': fold_idx,
                'epoch': epoch,
                'best_oracle_median': best_oracle,
            }, fold_outdir / "checkpoint_best_oracle.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            logger.info(f"Early stopping at epoch {epoch} (patience={config['patience']})")
            break

    logger.info(f"Fold {fold_idx}: Best oracle = {best_oracle:.2f} deg at epoch {best_epoch}")

    # Save training history
    with open(fold_outdir / 'training_history.json', 'w') as f:
        json.dump(train_history, f, indent=2)

    return {
        'fold_idx': fold_idx,
        'best_oracle_median': best_oracle,
        'best_epoch': best_epoch,
        'n_train': len(train_ids),
        'n_val': len(val_ids),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train asteroid pole prediction model (GeoHierK3Transformer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full 5-fold training (production recipe)
  python train_pole_model.py \\
      --csv-dir data/damit_csv_qf_ge_3 \\
      --spin-root data/damit_spins_complete \\
      --period-json data/periods.json \\
      --outdir checkpoints_new --folds 0,1,2,3,4

  # Single fold for quick testing
  python train_pole_model.py \\
      --csv-dir data/damit_csv_qf_ge_3 \\
      --spin-root data/damit_spins_complete \\
      --period-json data/periods.json \\
      --outdir test_output --folds 0 --epochs 5

See docs/USER_GUIDE.md for data preparation instructions.
        """
    )

    # Required data paths
    parser.add_argument('--csv-dir', type=Path, required=True,
                        help='Directory with DAMIT lightcurve CSV files')
    parser.add_argument('--spin-root', type=Path, required=True,
                        help='Directory with ground truth spin solution JSONs')
    parser.add_argument('--period-json', type=str, required=True,
                        help='JSON mapping asteroid IDs to periods in hours')
    parser.add_argument('--outdir', type=Path, default=Path('checkpoints_output'),
                        help='Output directory (default: checkpoints_output)')

    # Fold configuration
    parser.add_argument('--folds', type=str, default='0',
                        help='Comma-separated fold indices (default: 0)')
    parser.add_argument('--n-folds', type=int, default=5,
                        help='Total number of CV folds (default: 5)')
    parser.add_argument('--split-dir', type=Path, default=None,
                        help='Directory with pre-computed fold JSONs (optional)')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Max training epochs (default: 200)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--seed', type=int, default=777,
                        help='Random seed (default: 777)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')

    args = parser.parse_args()

    # Validate inputs
    if not args.csv_dir.is_dir():
        print(f"ERROR: CSV directory not found: {args.csv_dir}")
        sys.exit(1)
    if not args.spin_root.is_dir():
        print(f"ERROR: Spin root directory not found: {args.spin_root}")
        sys.exit(1)
    if not Path(args.period_json).is_file():
        print(f"ERROR: Period JSON not found: {args.period_json}")
        sys.exit(1)

    # Device check
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load object IDs and create splits
    object_ids = load_object_ids(args.csv_dir)
    logger.info(f"Found {len(object_ids)} asteroids in {args.csv_dir}")

    if args.split_dir is not None:
        folds_train, folds_val = [], []
        for i in range(args.n_folds):
            with open(args.split_dir / f"fold{i}_train.json") as f:
                folds_train.append(json.load(f))
            with open(args.split_dir / f"fold{i}_test.json") as f:
                folds_val.append(json.load(f))
    else:
        folds_train, folds_val = deterministic_fold_split(
            object_ids, n_folds=args.n_folds, seed=args.seed
        )

    fold_indices = [int(f) for f in args.folds.split(',')]

    # Training config (production defaults)
    config = {
        'device': args.device,
        'seed': args.seed,
        'epochs': args.epochs,
        'patience': args.patience,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'dropout': 0.1,
        'd_model': 128,
        'n_heads': 4,
        'n_layers_window': 4,
        'n_layers_cross': 0,
        'period_json': args.period_json,
        'min_gap_days': 30.0,
        # Loss hyperparameters (production values)
        'softmin_tau_deg': 5.0,
        'lambda_div': 0.5,
        'div_sigma_deg': 15.0,
        'lambda_var': 5.0,
        'lambda_sim': 2.0,
        'gap_tau_deg': 10.0,
        'ignore_near_ties_deg': 1.0,
    }

    # Save config
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Train requested folds
    all_results = []
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
        all_results.append(result)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info(f"{'='*60}")
    for r in all_results:
        logger.info(
            f"  Fold {r['fold_idx']}: oracle={r['best_oracle_median']:.2f} deg "
            f"(epoch {r['best_epoch']}, {r['n_train']} train / {r['n_val']} val)"
        )

    with open(outdir / 'training_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\nCheckpoints saved to: {outdir}/")


if __name__ == "__main__":
    main()
