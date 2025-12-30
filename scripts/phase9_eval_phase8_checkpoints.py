#!/usr/bin/env python3
"""
Phase 9: Canonical evaluation of Phase 8 checkpoints.

Evaluates all Phase 8 fold checkpoints using unified grid-MAP evaluator
(same as Phase 7) to confirm actual median error and identify any per-fold issues.

Usage:
    python scripts/phase9_eval_phase8_checkpoints.py \
        --root artifacts/phase8_seed5000 \
        --dataset datasets/pole_synth_multiepoch_v1 \
        --outdir artifacts/phase9_eval_phase8
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from pole_synth.model_epoch_attention_grid import EpochAttentionGridModel, generate_fibonacci_poles
from scripts.phase8_train_epoch_attention_grid import load_dataset, MultiEpochDataset, collate_fn
from pole_synth.losses import angular_distance_with_antipode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_fold(
    fold_idx: int,
    model_path: Path,
    dataloader: DataLoader,
    device: str,
) -> Dict:
    """
    Evaluate a single fold checkpoint using grid-MAP on 4096 pole grid.

    Returns:
        Dict with median, p90, mean, std, accuracy@25, per-sample errors
    """
    # Load model
    model = EpochAttentionGridModel(
        d_model=64,
        n_layers=2,
        n_heads=4,
        dim_feedforward=256,
        dropout=0.1,
        max_epochs=5,  # From v1 dataset shape
        token_dim=28,
        n_poles=4096,
    )

    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Fold {fold_idx}: Loaded checkpoint from {model_path}")
    except Exception as e:
        logger.error(f"Fold {fold_idx}: Failed to load checkpoint: {e}")
        return None

    model = model.to(device)
    model.eval()

    all_errors = []
    all_poles_true = []
    all_poles_pred = []

    with torch.no_grad():
        for batch_idx, (epochs_batch, masks_batch, poles_batch) in enumerate(dataloader):
            epochs_batch = epochs_batch.to(device)
            masks_batch = masks_batch.to(device)
            poles_batch = poles_batch.to(device)

            B = epochs_batch.shape[0]

            # Forward pass
            outputs = model.forward(epochs_batch, masks_batch)
            logp_grid = outputs["logp_grid"]  # (B, n_poles)

            # Predict best pole via argmax in logp space
            best_idx = torch.argmax(logp_grid, dim=1)  # (B,)
            poles_pred = model.pole_grid[best_idx]  # (B, 3)

            # Compute antipode-aware errors
            errors = angular_distance_with_antipode(poles_pred, poles_batch)  # (B,)

            all_errors.extend(errors.cpu().numpy())
            all_poles_true.extend(poles_batch.cpu().numpy())
            all_poles_pred.extend(poles_pred.cpu().numpy())

    all_errors = np.array(all_errors)
    results = {
        "fold": fold_idx,
        "n_samples": len(all_errors),
        "median_error_deg": float(np.median(all_errors)),
        "p90_error_deg": float(np.percentile(all_errors, 90)),
        "p95_error_deg": float(np.percentile(all_errors, 95)),
        "mean_error_deg": float(np.mean(all_errors)),
        "std_error_deg": float(np.std(all_errors)),
        "accuracy_at_10deg": float((all_errors <= 10.0).mean()),
        "accuracy_at_25deg": float((all_errors <= 25.0).mean()),
        "accuracy_at_45deg": float((all_errors <= 45.0).mean()),
        "per_sample_errors": all_errors.tolist(),
    }

    logger.info(
        f"Fold {fold_idx}: median={results['median_error_deg']:.2f}° "
        f"p90={results['p90_error_deg']:.2f}° acc@25°={results['accuracy_at_25deg']:.1%}"
    )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Canonical evaluation of Phase 8 checkpoints"
    )
    parser.add_argument("--root", required=True, help="Root directory of Phase 8 training (e.g., artifacts/phase8_seed5000)")
    parser.add_argument("--dataset", required=True, help="Path to dataset")
    parser.add_argument("--outdir", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")

    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("PHASE 9: CANONICAL EVALUATION OF PHASE 8 CHECKPOINTS")
    logger.info("=" * 80)
    logger.info(f"Root: {root}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Output: {outdir}")

    # Load dataset
    logger.info("Loading dataset...")
    epochs, poles, manifest = load_dataset(args.dataset)
    dataset = MultiEpochDataset(epochs, poles)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    logger.info(f"Loaded {len(dataset)} samples")

    # Evaluate each fold
    fold_results = []
    for fold_idx in range(5):
        fold_dir = root / f"fold{fold_idx}"
        model_path = fold_dir / "model_best.pt"

        if not model_path.exists():
            logger.warning(f"Fold {fold_idx}: Model not found at {model_path}, skipping")
            continue

        result = evaluate_fold(fold_idx, model_path, dataloader, args.device)
        if result is not None:
            fold_results.append(result)

    # Aggregate results
    all_medians = [r["median_error_deg"] for r in fold_results]
    all_p90s = [r["p90_error_deg"] for r in fold_results]
    all_accs = [r["accuracy_at_25deg"] for r in fold_results]

    summary = {
        "overall_median": float(np.median(all_medians)),
        "overall_p90": float(np.median(all_p90s)),
        "overall_mean_error": float(np.mean([r["mean_error_deg"] for r in fold_results])),
        "overall_accuracy_at_25deg": float(np.mean(all_accs)),
        "fold_results": fold_results,
        "timestamp": str(np.datetime64("now")),
    }

    # Save results
    with open(outdir / "phase8_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {outdir / 'phase8_eval_summary.json'}")

    # Generate markdown report
    report = f"""# Phase 9: Canonical Evaluation of Phase 8 Checkpoints

**Date**: {summary['timestamp']}
**Root**: {args.root}
**Dataset**: {args.dataset}

## Overall Results

| Metric | Value |
|--------|-------|
| **Median Error** | {summary['overall_median']:.2f}° |
| **P90 Error** | {summary['overall_p90']:.2f}° |
| **Mean Error** | {summary['overall_mean_error']:.2f}° |
| **Accuracy @25°** | {summary['overall_accuracy_at_25deg']:.1%} |
| **N Folds** | {len(fold_results)} |

## Per-Fold Results

| Fold | Median | P90 | P95 | Mean | Acc@25° | N Samples |
|------|--------|-----|-----|------|---------|-----------|
"""

    for r in fold_results:
        report += (
            f"| {r['fold']} | {r['median_error_deg']:.2f}° | {r['p90_error_deg']:.2f}° | "
            f"{r['p95_error_deg']:.2f}° | {r['mean_error_deg']:.2f}° | "
            f"{r['accuracy_at_25deg']:.1%} | {r['n_samples']} |\n"
        )

    report += f"""

## Diagnosis

- **Oracle Median on V1**: 16.2°
- **Phase 8 Median**: {summary['overall_median']:.2f}°
- **Gap**: {summary['overall_median'] - 16.2:.1f}°
- **Phase 7 Baseline**: ~50.68°

### Interpretation

If median ≤ 30°: Model learns meaningful signal, issue is regularization/optimization.
If median > 45°: Pipeline likely broken (targets/features/symmetry mismatch).
If median ≈ oracle (≤ 20°): Model working, needs minimal tuning.

**Current Status**: {'🟢 Model learning' if summary['overall_median'] <= 35 else '🟡 Mixed' if summary['overall_median'] <= 50 else '🔴 Pipeline broken'}

## Next Step

Run overfit sanity suite (phase9_overfit_sanity.py) to determine if model can fit single objects.
"""

    with open(outdir / "phase8_eval_report.md", "w") as f:
        f.write(report)
    logger.info(f"Saved report to {outdir / 'phase8_eval_report.md'}")

    logger.info("=" * 80)
    logger.info(f"Overall Median: {summary['overall_median']:.2f}°")
    logger.info(f"Overall Accuracy @25°: {summary['overall_accuracy_at_25deg']:.1%}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
