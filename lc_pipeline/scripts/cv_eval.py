"""Cross-fold evaluation script for CV177 QF>=3 5-fold cross-validation.

This script:
1. Loads trained fold checkpoints (fold_0 through fold_4)
2. Evaluates oracle error on validation sets
3. Aggregates metrics across 5 folds
4. Generates comprehensive evaluation report

Supports two checkpoint layouts:
- Fold directories: outdir/fold_0/checkpoint_best_oracle.pt (from train_k3.py)
- Flat checkpoints: outdir/fold_0.pt (shipped production checkpoints)

Expected Results (production model, CV177 QF>=3):
- Fold 0: 18.55 deg | Fold 1: 15.28 deg | Fold 2: 20.65 deg
- Fold 3: 21.80 deg | Fold 4: 19.29 deg
- Mean: 19.12 +/- 2.22 deg (asteroid-level)

Usage (shipped checkpoints):
    python -m lc_pipeline.scripts.cv_eval \
        --csv-dir data/damit_csv_qf_ge_3 \
        --spin-root data/damit_spins_complete \
        --outdir lc_pipeline/checkpoints \
        --checkpoint-pattern "fold_{}.pt" \
        --period-json data/damit_periods.json \
        --folds 0,1,2,3,4 \
        --n-folds 5 \
        --device cuda

Usage (training output directories):
    python -m lc_pipeline.scripts.cv_eval \
        --csv-dir data/damit_csv_qf_ge_3 \
        --spin-root data/damit_spins_complete \
        --outdir /tmp/train_output \
        --period-json data/damit_periods.json \
        --folds 0,1,2,3,4 \
        --n-folds 5 \
        --device cuda
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from lc_pipeline.data.single_epoch_dataset import SingleEpochDataset, collate_fn as se_collate_fn
from lc_pipeline.models.geo_hier_k3_transformer import GeoHierK3Transformer, load_checkpoint
from lc_pipeline.evaluation.eval_axisnet import evaluate_fold, aggregate_folds

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_object_ids(csv_dir: Path) -> List[str]:
    """Load list of object IDs from CSV directory."""
    csv_dir = Path(csv_dir)
    csv_files = sorted(csv_dir.glob("*.csv"))
    object_ids = [f.stem for f in csv_files]
    return object_ids


def deterministic_fold_split(object_ids: List[str], n_folds: int = 5, seed: int = 1337):
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


def find_best_checkpoint(fold_dir: Path) -> Optional[Path]:
    """Find the best checkpoint in fold directory (lowest val loss)."""
    fold_dir = Path(fold_dir)

    # Prefer best oracle checkpoint
    best_oracle = fold_dir / 'checkpoint_best_oracle.pt'
    if best_oracle.exists():
        logger.info(f"Found best oracle checkpoint: {best_oracle}")
        return best_oracle

    # Fallback: look at training history to find best epoch
    history_file = fold_dir / 'training_history.json'
    if history_file.exists():
        with open(history_file) as f:
            history = json.load(f)

        if history['val_history']:
            best_epoch = min(history['val_history'], key=lambda x: x['metrics']['oracle_median_deg'] or float('inf'))
            best_epoch_idx = best_epoch['epoch']

            checkpoint_file = fold_dir / f"checkpoint_epoch_{best_epoch_idx}.pt"
            if checkpoint_file.exists():
                logger.info(f"Found best checkpoint: {checkpoint_file} (epoch {best_epoch_idx})")
                return checkpoint_file

    # Fallback: look for any checkpoint
    checkpoints = sorted(fold_dir.glob("checkpoint_*.pt"))
    if checkpoints:
        logger.warning(f"Using latest checkpoint: {checkpoints[-1]}")
        return checkpoints[-1]

    return None


def evaluate_fold_checkpoint(
    fold_idx: int,
    checkpoint_path: Path,
    val_ids: List[str],
    csv_dir: Path,
    spin_root: Optional[Path],
    device: str = 'cuda',
    seed: int = 1337,
    period_json: Optional[str] = None,
) -> Dict:
    """Evaluate one fold checkpoint on validation set."""
    logger.info(f"\n{'='*60}")
    logger.info(f"EVALUATING FOLD {fold_idx}")
    logger.info(f"{'='*60}\n")

    # Load model and config
    model, config = load_checkpoint(str(checkpoint_path))
    model.to(device)
    logger.info(f"Loaded model from {checkpoint_path}")

    # Create validation dataset using SingleEpochDataset (matches training)
    val_ds = SingleEpochDataset(
        csv_dir=csv_dir,
        spin_root=spin_root,
        object_ids=val_ids,
        min_gap_days=30.0,
        use_geometry=False,
        period_json=period_json,
    )

    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, collate_fn=se_collate_fn,
        num_workers=0, pin_memory=True,
    )

    # Evaluate on val set
    logger.info(f"Evaluating on {len(val_ids)} validation objects...")
    fold_eval = evaluate_fold(model, val_loader, device=device)
    logger.info(f"Oracle median: {fold_eval['metrics']['oracle_median_deg']:.2f} deg")
    logger.info(f"Gap median: {fold_eval['metrics']['gap_median_deg']:.2f} deg")

    selector_acc = fold_eval['metrics'].get('selector_accuracy')
    if selector_acc is not None:
        logger.info(f"Selector accuracy: {selector_acc:.3f}")

    return {
        'fold_idx': fold_idx,
        'n_val': len(val_ids),
        'eval_results': fold_eval,
        'checkpoint': str(checkpoint_path),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate K=3 pole model across CV folds')

    parser.add_argument('--outdir', type=Path, required=True, help='Directory with checkpoints (fold dirs or flat files)')
    parser.add_argument('--csv-dir', type=Path, required=True, help='Path to DAMIT CSV directory')
    parser.add_argument('--spin-root', type=Path, default=None, help='Path to spin solutions directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4', help='Comma-separated fold indices')
    parser.add_argument('--n-folds', type=int, default=5, help='Total number of folds')
    parser.add_argument('--checkpoint-pattern', type=str, default=None,
                        help='Pattern for flat checkpoint files, e.g. "fold_{}.pt". '
                             'The {} is replaced with fold index. If not provided, '
                             'looks for fold subdirectories with training history.')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for fold splits. If not provided, reads from '
                             'config.json or first checkpoint.')
    parser.add_argument('--split-dir', type=Path, default=None,
                        help='Directory with pre-computed fold JSONs (fold{i}_train.json / fold{i}_test.json). '
                             'When provided, overrides deterministic_fold_split().')
    parser.add_argument('--period-json', type=str, default=None,
                        help='Path to JSON with DAMIT periods per asteroid. '
                             'Required for period-aware evaluation (features 9-12).')

    args = parser.parse_args()

    outdir = Path(args.outdir)

    # Determine seed for fold splits (priority: CLI arg > config.json > checkpoint > default)
    seed = args.seed
    if seed is None:
        config_file = outdir / 'config.json'
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            seed = config.get('seed', 777)
            logger.info(f"Loaded seed={seed} from {config_file}")
        else:
            # Try to read seed from first checkpoint
            fold_indices = [int(f) for f in args.folds.split(',')]
            for fi in fold_indices:
                if args.checkpoint_pattern:
                    ckpt_path = outdir / args.checkpoint_pattern.format(fi)
                else:
                    ckpt_path = outdir / f"fold_{fi}" / "checkpoint_best_oracle.pt"
                if ckpt_path.exists():
                    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
                    ckpt_config = ckpt.get('config', {})
                    seed = ckpt_config.get('seed', 777)
                    logger.info(f"Loaded seed={seed} from checkpoint {ckpt_path}")
                    break
            if seed is None:
                seed = 777
                logger.info(f"No config found, using default seed={seed}")

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
        folds_train, folds_val = deterministic_fold_split(
            object_ids, n_folds=args.n_folds, seed=seed
        )

    # Parse fold indices
    fold_indices = [int(f) for f in args.folds.split(',')]

    # Evaluate each fold
    fold_eval_results = []
    for fold_idx in fold_indices:
        # Resolve checkpoint path
        if args.checkpoint_pattern:
            # Flat checkpoint mode: outdir/fold_0.pt
            checkpoint_path = outdir / args.checkpoint_pattern.format(fold_idx)
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                continue
        else:
            # Fold directory mode: outdir/fold_0/checkpoint_best_oracle.pt
            fold_dir = outdir / f"fold_{fold_idx}"
            if not fold_dir.exists():
                logger.warning(f"Fold directory not found: {fold_dir}")
                continue

            checkpoint_path = find_best_checkpoint(fold_dir)
            if checkpoint_path is None:
                logger.warning(f"No checkpoint found for fold {fold_idx}")
                continue

        # Evaluate fold
        result = evaluate_fold_checkpoint(
            fold_idx=fold_idx,
            checkpoint_path=checkpoint_path,
            val_ids=folds_val[fold_idx],
            csv_dir=args.csv_dir,
            spin_root=args.spin_root,
            device=args.device,
            seed=seed,
            period_json=args.period_json,
        )
        fold_eval_results.append(result)

    # Aggregate across folds
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATING RESULTS ACROSS FOLDS")
    logger.info(f"{'='*60}\n")

    if fold_eval_results:
        fold_only_results = [r['eval_results'] for r in fold_eval_results]
        aggregate_metrics = aggregate_folds(fold_only_results)

        logger.info(f"Oracle median (all folds): {aggregate_metrics['oracle_median_deg']:.2f} deg")
        logger.info(f"Oracle mean (all folds): {aggregate_metrics['oracle_mean_deg']:.2f} deg")
        logger.info(f"Quality median (all folds): {aggregate_metrics['quality_median_deg']:.2f} deg")
        logger.info(f"Gap median (all folds): {aggregate_metrics['gap_median_deg']:.2f} deg")
        logger.info(f"Selector accuracy (all folds): {aggregate_metrics['selector_accuracy']:.3f}")
        logger.info(f"Total objects evaluated: {aggregate_metrics['total_objects']}")

        # Per-fold summary
        logger.info(f"\nPer-fold oracle median:")
        for r in fold_eval_results:
            fold_idx = r['fold_idx']
            oracle_med = r['eval_results']['metrics']['oracle_median_deg']
            logger.info(f"  Fold {fold_idx}: {oracle_med:.2f} deg")

        # Save evaluation report
        eval_report = {
            'aggregate_metrics': aggregate_metrics,
            'fold_results': fold_eval_results,
        }

        eval_report_file = outdir / 'evaluation_report.json'
        with open(eval_report_file, 'w') as f:
            json.dump(eval_report, f, indent=2)
        logger.info(f"\nSaved evaluation report to {eval_report_file}")

    logger.info(f"\n{'='*60}")
    logger.info("Evaluation complete!")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
