#!/usr/bin/env python3
"""
Phase 29I: One-Command Paper Run.

Runs complete CV evaluation with learned selector and produces paper-ready
markdown report and JSON artifact.

Usage:
    python scripts/phase29i_paper_run.py \\
        --csv-dir DAMIT_csv_high \\
        --artifacts-root artifacts/phase29g_cv \\
        --selector-root artifacts/phase29h_selectors \\
        --folds 0,1,2 \\
        --outdir results/paper_run \\
        --device cuda

Output:
    - outdir/paper_results.json
    - outdir/paper_report.md
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from pole_synth.version import __version__, get_version_info
from pole_synth.reporting import (
    build_paper_results_json,
    build_paper_report,
    save_paper_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_inputs(args) -> List[str]:
    """Validate input paths exist. Returns list of errors."""
    errors = []

    if not args.csv_dir.exists():
        errors.append(f"CSV directory not found: {args.csv_dir}")

    if not args.artifacts_root.exists():
        errors.append(f"Artifacts root not found: {args.artifacts_root}")

    # Check fold directories exist
    for fold in args.folds:
        fold_dir = args.artifacts_root / f"fold{fold}"
        if not fold_dir.exists():
            errors.append(f"Fold directory not found: {fold_dir}")
        else:
            # Check for model checkpoint
            model_path = fold_dir / "model_best.pt"
            if not model_path.exists():
                errors.append(f"Model checkpoint not found: {model_path}")

    # Check selector if not training
    if not args.train_selector and args.selector_root:
        if not args.selector_root.exists():
            errors.append(f"Selector root not found: {args.selector_root}")

    return errors


def load_model_for_fold(
    fold_dir: Path,
    device: torch.device,
    d_model: int = 128,
    hidden_dim: int = 256,
    n_layers: int = 2,
    dropout: float = 0.1,
    pool_mode: str = "attention",
) -> Tuple[Any, int]:
    """Load K=2 model for a fold."""
    from pole_synth.model_phase15_pole_regressor import MultiHypothesisPoleRegressor

    model_path = fold_dir / "model_best.pt"

    # Detect K from checkpoint
    state = torch.load(model_path, map_location='cpu', weights_only=True)
    K = 2  # Default
    for key in state.keys():
        if 'pole_head' in key and 'weight' in key:
            output_dim = state[key].shape[0]
            if output_dim % 3 == 0:
                K = output_dim // 3
            break

    model = MultiHypothesisPoleRegressor(
        token_dim=9,
        d_model=d_model,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        pool_mode=pool_mode,
        n_hypotheses=K,
    )
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    return model, K


def load_or_train_selector(
    fold: int,
    fold_dir: Path,
    selector_root: Optional[Path],
    train_selector: bool,
    evaluator: Any,
    model: Any,
    objects: List,
    object_ids: List[str],
    spin_db: Optional[Dict],
    device: str,
    label_mode: str,
    seed: int,
) -> Optional[Any]:
    """Load existing selector or train new one."""
    from pole_synth.selector_model import HypothesisSelector, SelectorTrainer, load_selector, save_selector
    from pole_synth.selector_features import (
        build_selector_features,
        compute_pseudo_labels_canonical,
        compute_pseudo_labels_set,
        get_feature_dim,
    )

    # Try to load existing selector
    if selector_root is not None:
        selector_path = selector_root / f"fold{fold}" / "selector_best.pt"
        if selector_path.exists():
            logger.info(f"Loading selector from {selector_path}")
            return load_selector(selector_path, device=device)

        # Also try global selector
        global_path = selector_root / "selector_best.pt"
        if global_path.exists():
            logger.info(f"Loading global selector from {global_path}")
            return load_selector(global_path, device=device)

    if not train_selector:
        logger.warning(f"No selector found for fold {fold}, will use heuristic only")
        return None

    # Train new selector
    logger.info(f"Training selector for fold {fold}")

    # Compute features and labels
    K = 2
    features_list = []
    labels_list = []

    for obj_idx in range(len(objects)):
        try:
            result = evaluator.predict_object_hypotheses(
                model=model,
                obj_idx=obj_idx,
                device=device,
                batch_size=64,
            )
            hyp_xyz = result['hyp_xyz']
            window_preds_xyz = result['window_preds_xyz']

            features = build_selector_features(window_preds_xyz, hyp_xyz, normalize=True)

            # Get target for pseudo-label
            obj = objects[obj_idx]
            target_xyz = obj.get('pole')
            if target_xyz is None:
                target_xyz = obj.get('pole_ecl_xyz')
            if target_xyz is None:
                continue
            target_xyz = np.array(target_xyz)

            # Compute pseudo-label
            object_id = object_ids[obj_idx]
            asteroid_id = object_id.split('_')[1] if '_' in object_id else object_id
            spin_poles = None
            if spin_db is not None:
                spin_sol = spin_db.get(asteroid_id)
                if spin_sol is not None and hasattr(spin_sol, 'poles_ecl_xyz'):
                    spin_poles = spin_sol.poles_ecl_xyz

            if label_mode == 'canonical':
                label = compute_pseudo_labels_canonical(hyp_xyz, target_xyz)
            elif label_mode == 'set' and spin_poles is not None:
                label = compute_pseudo_labels_set(hyp_xyz, spin_poles)
            elif label_mode == 'hybrid':
                if spin_poles is not None and len(spin_poles) > 1:
                    label = compute_pseudo_labels_set(hyp_xyz, spin_poles)
                else:
                    label = compute_pseudo_labels_canonical(hyp_xyz, target_xyz)
            else:
                label = compute_pseudo_labels_canonical(hyp_xyz, target_xyz)

            features_list.append(features)
            labels_list.append(label)

        except Exception as e:
            continue

    if len(features_list) < 10:
        logger.warning(f"Not enough data for selector training ({len(features_list)} samples)")
        return None

    features = np.array(features_list)
    labels = np.array(labels_list)

    # Split train/val
    np.random.seed(seed)
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    n_val = int(len(features) * 0.2)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_features = features[train_idx]
    train_labels = labels[train_idx]
    val_features = features[val_idx]
    val_labels = labels[val_idx]

    # Create and train selector
    input_dim = get_feature_dim(K)
    selector = HypothesisSelector(
        input_dim=input_dim,
        hidden_dim=32,
        n_hypotheses=K,
        dropout=0.1,
    )

    trainer = SelectorTrainer(model=selector, lr=1e-3, device=device)
    trainer.train(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        n_epochs=100,
        batch_size=32,
        patience=20,
        verbose=False,
    )

    # Save selector
    if selector_root is not None:
        save_dir = selector_root / f"fold{fold}"
        save_dir.mkdir(parents=True, exist_ok=True)
        save_selector(selector, save_dir / "selector_best.pt", save_dir / "selector_config.json")
        logger.info(f"Saved selector to {save_dir}")

    return selector


def evaluate_fold(
    fold: int,
    args,
    spin_db: Optional[Dict],
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate a single fold with all selectors."""
    from pole_synth.damit_real_multiepoch_dataset import (
        FrozenWindowEvaluator,
        load_damit_dataset,
    )
    from pole_synth.inference import predict_pole_selected
    from pole_synth.ambiguity_metrics import (
        angular_sep_deg,
        get_object_label,
    )
    from pole_synth.selector_features import get_feature_dim

    fold_dir = args.artifacts_root / f"fold{fold}"
    logger.info(f"Evaluating fold {fold} from {fold_dir}")

    # Load objects
    objects, object_ids = load_damit_dataset(args.csv_dir)

    # Load model
    model, K = load_model_for_fold(fold_dir, device)
    logger.info(f"Loaded K={K} model")

    # Create evaluator
    evaluator = FrozenWindowEvaluator(
        objects,
        n_windows=256,
        n_epochs=5,
        n_tokens=64,
        seed=args.seed + fold,
    )

    # Load or train selector
    selector = load_or_train_selector(
        fold=fold,
        fold_dir=fold_dir,
        selector_root=args.selector_root,
        train_selector=args.train_selector,
        evaluator=evaluator,
        model=model,
        objects=objects,
        object_ids=object_ids,
        spin_db=spin_db,
        device=str(device),
        label_mode=args.label_mode,
        seed=args.seed,
    )

    # Evaluate all objects
    per_object = []

    for obj_idx in range(len(objects)):
        obj = objects[obj_idx]
        object_id = object_ids[obj_idx]

        # Get ground truth
        target_xyz = obj.get('pole')
        if target_xyz is None:
            target_xyz = obj.get('pole_ecl_xyz')
        if target_xyz is None:
            continue
        target_xyz = np.array(target_xyz)

        # Get spin solutions
        asteroid_id = object_id.split('_')[1] if '_' in object_id else object_id
        spin_poles = None
        if spin_db is not None:
            spin_sol = spin_db.get(asteroid_id)
            if spin_sol is not None and hasattr(spin_sol, 'poles_ecl_xyz'):
                spin_poles = spin_sol.poles_ecl_xyz

        label = get_object_label(spin_poles, 15.0) if spin_poles is not None else 'unique'

        try:
            # Agreement selector
            result_agreement = predict_pole_selected(
                model, evaluator, obj_idx, str(device), 64,
                selector_mode='agreement',
            )

            hyp_xyz = result_agreement['hyp_xyz']

            # Oracle selection
            can_errors = [angular_sep_deg(hyp_xyz[k], target_xyz) for k in range(K)]
            oracle_idx = int(np.argmin(can_errors))
            can_err_oracle = can_errors[oracle_idx]

            # Set errors
            if spin_poles is not None:
                set_errors = []
                for k in range(K):
                    min_dist = min(angular_sep_deg(hyp_xyz[k], p) for p in spin_poles)
                    set_errors.append(min_dist)
                set_err_oracle = min(set_errors)
            else:
                set_errors = can_errors
                set_err_oracle = can_err_oracle

            record = {
                'object_id': object_id,
                'label': label,
                # Oracle
                'can_err_oracle': can_err_oracle,
                'set_err_oracle': set_err_oracle,
                'oracle_idx': oracle_idx,
                # Agreement
                'can_err_agreement': can_errors[result_agreement['selected_idx']],
                'set_err_agreement': set_errors[result_agreement['selected_idx']],
                'agreement_idx': result_agreement['selected_idx'],
                # Naive0
                'can_err_naive0': can_errors[0],
                'set_err_naive0': set_errors[0],
                # Inter-hypothesis
                'interhyp_angle': angular_sep_deg(hyp_xyz[0], hyp_xyz[1]) if K == 2 else 0,
            }

            # Learned selector
            if selector is not None:
                result_learned = predict_pole_selected(
                    model, evaluator, obj_idx, str(device), 64,
                    selector_mode='learned',
                    selector_model=selector,
                )
                record['can_err_learned'] = can_errors[result_learned['selected_idx']]
                record['set_err_learned'] = set_errors[result_learned['selected_idx']]
                record['learned_idx'] = result_learned['selected_idx']
                record['learned_confidence'] = result_learned.get('confidence', 1.0)

            per_object.append(record)

        except Exception as e:
            logger.debug(f"Error evaluating {object_id}: {e}")
            continue

    # Aggregate results
    return aggregate_per_object_to_fold_results(per_object, selector is not None)


def aggregate_per_object_to_fold_results(
    per_object: List[Dict],
    has_learned: bool,
) -> Dict[str, Any]:
    """Aggregate per-object results into fold-level metrics."""
    if not per_object:
        return {}

    results = {}

    # Helper to compute metrics
    def compute_metrics(errors: List[float]) -> Dict[str, float]:
        if not errors:
            return {}
        return {
            'set_err_mean': float(np.mean(errors)),
            'set_err_median': float(np.median(errors)),
            'set_err_std': float(np.std(errors)),
        }

    # Overall metrics
    for key, err_key in [
        ('oracle_k2', 'set_err_oracle'),
        ('selected_agreement_k2', 'set_err_agreement'),
        ('naive0_k2', 'set_err_naive0'),
    ]:
        errors = [r[err_key] for r in per_object if err_key in r]
        can_errors = [r[err_key.replace('set_', 'can_')] for r in per_object if err_key.replace('set_', 'can_') in r]

        results[key] = {
            'overall': {
                'set_err_mean': float(np.mean(errors)) if errors else 0,
                'set_err_median': float(np.median(errors)) if errors else 0,
                'can_err_mean': float(np.mean(can_errors)) if can_errors else 0,
                'can_err_median': float(np.median(can_errors)) if can_errors else 0,
            },
            'by_label': {},
        }

        # By label
        for label in ['unique', 'multi_close', 'multi_far']:
            label_objs = [r for r in per_object if r.get('label') == label]
            if label_objs:
                label_errors = [r[err_key] for r in label_objs if err_key in r]
                label_can = [r[err_key.replace('set_', 'can_')] for r in label_objs if err_key.replace('set_', 'can_') in r]
                results[key]['by_label'][label] = {
                    'set_err_mean': float(np.mean(label_errors)) if label_errors else 0,
                    'set_err_median': float(np.median(label_errors)) if label_errors else 0,
                    'can_err_mean': float(np.mean(label_can)) if label_can else 0,
                    'can_err_median': float(np.median(label_can)) if label_can else 0,
                    'count': len(label_objs),
                }

    # Agreement selector accuracy
    oracle_indices = [r['oracle_idx'] for r in per_object]
    agreement_indices = [r['agreement_idx'] for r in per_object]
    results['selected_agreement_k2']['selector_accuracy'] = float(np.mean([
        a == o for a, o in zip(agreement_indices, oracle_indices)
    ]))

    # Learned selector
    if has_learned:
        learned_errors = [r['set_err_learned'] for r in per_object if 'set_err_learned' in r]
        learned_can = [r['can_err_learned'] for r in per_object if 'can_err_learned' in r]

        results['selected_learned_k2'] = {
            'overall': {
                'set_err_mean': float(np.mean(learned_errors)) if learned_errors else 0,
                'set_err_median': float(np.median(learned_errors)) if learned_errors else 0,
                'can_err_mean': float(np.mean(learned_can)) if learned_can else 0,
                'can_err_median': float(np.median(learned_can)) if learned_can else 0,
            },
            'by_label': {},
        }

        # By label
        for label in ['unique', 'multi_close', 'multi_far']:
            label_objs = [r for r in per_object if r.get('label') == label and 'set_err_learned' in r]
            if label_objs:
                results['selected_learned_k2']['by_label'][label] = {
                    'set_err_mean': float(np.mean([r['set_err_learned'] for r in label_objs])),
                    'set_err_median': float(np.median([r['set_err_learned'] for r in label_objs])),
                    'can_err_mean': float(np.mean([r['can_err_learned'] for r in label_objs])),
                    'can_err_median': float(np.median([r['can_err_learned'] for r in label_objs])),
                    'count': len(label_objs),
                }

        # Selector accuracy
        learned_indices = [r['learned_idx'] for r in per_object if 'learned_idx' in r]
        if learned_indices:
            results['selected_learned_k2']['selector_accuracy'] = float(np.mean([
                l == o for l, o in zip(learned_indices, oracle_indices[:len(learned_indices)])
            ]))

        # Calibration
        objs_with_conf = [r for r in per_object if 'learned_confidence' in r]
        if len(objs_with_conf) >= 5:
            confidences = np.array([r['learned_confidence'] for r in objs_with_conf])
            gaps = np.array([r['set_err_learned'] - r['set_err_oracle'] for r in objs_with_conf])
            correct = np.array([r['learned_idx'] == r['oracle_idx'] for r in objs_with_conf])

            n_bins = 5
            bin_edges = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
            bin_edges[-1] += 0.01

            bins = []
            for i in range(n_bins):
                mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
                if mask.sum() > 0:
                    bins.append({
                        'bin_idx': i,
                        'conf_low': float(bin_edges[i]),
                        'conf_high': float(bin_edges[i + 1]),
                        'n_objects': int(mask.sum()),
                        'accuracy': float(correct[mask].mean()),
                        'gap_mean': float(gaps[mask].mean()),
                    })

            results['calibration'] = {'bins': bins}
        else:
            results['calibration'] = {'bins': []}

    # Diagnostics
    interhyp_angles = [r['interhyp_angle'] for r in per_object if 'interhyp_angle' in r]
    collapsed = [r for r in per_object if r.get('interhyp_angle', 90) < 5]
    complementary = [r for r in per_object
                     if r.get('set_err_oracle') is not None
                     and r.get('set_err_naive0') is not None
                     and (r['set_err_naive0'] - r['set_err_oracle']) >= 5]

    results['diagnostics'] = {
        'collapse_rate': len(collapsed) / len(per_object) if per_object else 0,
        'complementarity_rate': len(complementary) / len(per_object) if per_object else 0,
        'mean_interhyp_angle': float(np.mean(interhyp_angles)) if interhyp_angles else 0,
    }

    results['n_objects'] = len(per_object)
    results['per_object'] = per_object

    return results


def validate_output_schema(results: Dict[str, Any]) -> List[str]:
    """
    Validate output JSON has required schema keys.

    Returns list of missing/invalid items (empty if valid).
    """
    issues = []

    # Top-level required keys
    required_top = ['version', 'version_info', 'timestamp', 'config', 'commands', 'n_folds', 'folds', 'aggregates']
    for key in required_top:
        if key not in results:
            issues.append(f"Missing top-level key: {key}")

    # Aggregates structure
    agg = results.get('aggregates', {})
    required_agg = ['oracle_k2', 'selected_agreement_k2', 'naive0_k2']
    for key in required_agg:
        if key not in agg:
            issues.append(f"Missing aggregate key: {key}")
        elif 'overall' not in agg.get(key, {}):
            issues.append(f"Missing 'overall' in aggregates.{key}")

    # Calibration (if learned selector present)
    if 'selected_learned_k2' in agg:
        if 'calibration' not in agg:
            issues.append("Missing calibration data for learned selector")

    # Diagnostics
    if 'diagnostics' not in agg:
        issues.append("Missing diagnostics in aggregates")
    else:
        diag = agg['diagnostics']
        for metric in ['collapse_rate_mean', 'complementarity_rate_mean', 'mean_interhyp_angle_mean']:
            if metric not in diag:
                issues.append(f"Missing diagnostics metric: {metric}")

    # Commands
    if not results.get('commands'):
        issues.append("Commands list is empty")

    return issues


def build_command_string(args) -> str:
    """Build the command string that was executed."""
    cmd_parts = ["python scripts/phase29i_paper_run.py"]
    cmd_parts.append(f"--csv-dir {args.csv_dir}")
    cmd_parts.append(f"--artifacts-root {args.artifacts_root}")
    if args.selector_root:
        cmd_parts.append(f"--selector-root {args.selector_root}")
    cmd_parts.append(f"--folds {','.join(map(str, args.folds))}")
    cmd_parts.append(f"--device {args.device}")
    cmd_parts.append(f"--seed {args.seed}")
    cmd_parts.append(f"--selector-mode {args.selector_mode}")
    cmd_parts.append(f"--label-mode {args.label_mode}")
    if args.train_selector:
        cmd_parts.append("--train-selector")
    cmd_parts.append(f"--outdir {args.outdir}")
    return " \\\n    ".join(cmd_parts)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 29I: One-Command Paper Run"
    )

    # Data arguments
    parser.add_argument("--csv-dir", type=Path, required=True,
                        help="Directory with DAMIT CSV files")
    parser.add_argument("--period-cache", type=Path, default=None,
                        help="Path to period cache JSON (optional)")
    parser.add_argument("--spin-root", type=Path, default=None,
                        help="Path to spin solutions cache JSON (optional)")

    # Model/selector paths
    parser.add_argument("--artifacts-root", type=Path, required=True,
                        help="Root directory with fold subdirs containing K=2 models")
    parser.add_argument("--selector-root", type=Path, default=None,
                        help="Root directory with selector checkpoints")

    # Fold configuration
    parser.add_argument("--folds", type=str, default="0,1,2",
                        help="Comma-separated fold indices")

    # Selector configuration
    parser.add_argument("--selector-mode", type=str, default="learned",
                        choices=["learned", "agreement"],
                        help="Primary selector mode for results")
    parser.add_argument("--label-mode", type=str, default="hybrid",
                        choices=["canonical", "set", "hybrid"],
                        help="Pseudo-label mode for selector training")
    parser.add_argument("--train-selector", action="store_true",
                        help="Train selectors if not found")

    # Output
    parser.add_argument("--outdir", type=Path, required=True,
                        help="Output directory for results")

    # Misc
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=99999)

    args = parser.parse_args()

    # Parse folds
    args.folds = [int(f.strip()) for f in args.folds.split(',')]

    # Setup
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Version: {__version__}")

    # Validate inputs
    errors = validate_inputs(args)
    if errors:
        for err in errors:
            logger.error(err)
        sys.exit(1)

    args.outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PHASE 29I: ONE-COMMAND PAPER RUN")
    logger.info("=" * 70)

    # Load spin solutions
    spin_db = None
    spin_cache_path = args.spin_root or Path("results/spin_solutions_damit.json")
    if spin_cache_path.exists():
        from pole_synth.spin_solutions import load_spin_solution_cache
        logger.info(f"Loading spin cache from {spin_cache_path}")
        spin_db = load_spin_solution_cache(spin_cache_path)
        logger.info(f"Spin cache: {len(spin_db)} asteroids")

    # Evaluate each fold
    fold_results = {}
    for fold in args.folds:
        logger.info(f"\n{'='*70}")
        logger.info(f"FOLD {fold}")
        logger.info(f"{'='*70}")

        fold_result = evaluate_fold(fold, args, spin_db, device)
        fold_results[str(fold)] = fold_result

        if fold_result:
            n_obj = fold_result.get('n_objects', 0)
            oracle_mean = fold_result.get('oracle_k2', {}).get('overall', {}).get('set_err_mean', 0)
            logger.info(f"Fold {fold}: {n_obj} objects, Oracle@2 = {oracle_mean:.2f}°")

    # Build results
    command = build_command_string(args)
    config = {
        'csv_dir': str(args.csv_dir),
        'artifacts_root': str(args.artifacts_root),
        'selector_root': str(args.selector_root) if args.selector_root else None,
        'folds': args.folds,
        'selector_mode': args.selector_mode,
        'label_mode': args.label_mode,
        'train_selector': args.train_selector,
        'device': args.device,
        'seed': args.seed,
    }

    results = build_paper_results_json(
        fold_results=fold_results,
        commands=[command],
        config=config,
    )

    # Validate output schema
    schema_issues = validate_output_schema(results)
    if schema_issues:
        logger.warning("Schema validation issues (non-fatal):")
        for issue in schema_issues:
            logger.warning(f"  - {issue}")

    # Save results
    json_path = args.outdir / "paper_results.json"
    md_path = args.outdir / "paper_report.md"
    save_paper_results(results, json_path, md_path)

    logger.info("")
    logger.info("=" * 70)
    logger.info("PAPER RUN COMPLETE")
    logger.info("=" * 70)
    logger.info(f"JSON: {json_path}")
    logger.info(f"Report: {md_path}")

    # Print summary
    aggregates = results.get('aggregates', {})
    oracle_mean = aggregates.get('oracle_k2', {}).get('overall', {}).get('set_err_mean_mean', 0)
    oracle_std = aggregates.get('oracle_k2', {}).get('overall', {}).get('set_err_mean_std', 0)
    logger.info(f"Oracle@2: {oracle_mean:.2f}±{oracle_std:.2f}°")

    if 'selected_learned_k2' in aggregates:
        learned_mean = aggregates['selected_learned_k2']['overall'].get('set_err_mean_mean', 0)
        learned_std = aggregates['selected_learned_k2']['overall'].get('set_err_mean_std', 0)
        logger.info(f"Selected@2 (Learned): {learned_mean:.2f}±{learned_std:.2f}°")

    agreement_mean = aggregates.get('selected_agreement_k2', {}).get('overall', {}).get('set_err_mean_mean', 0)
    agreement_std = aggregates.get('selected_agreement_k2', {}).get('overall', {}).get('set_err_mean_std', 0)
    logger.info(f"Selected@2 (Agreement): {agreement_mean:.2f}±{agreement_std:.2f}°")


if __name__ == "__main__":
    main()
