"""
Phase 29C: Unified Ambiguity-Aware Metrics Reporter.

Evaluates models with both canonical and set metrics, stratified by ambiguity class.
Uses existing FrozenWindowEvaluator + best_agreement aggregation for consistency
with Phase 28B protocol.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from pole_synth.ambiguity import (
    AmbiguityInfo,
    angular_sep_deg,
    build_ambiguity_index,
    extract_asteroid_id,
)
from pole_synth.damit_real_multiepoch_dataset import (
    FrozenWindowEvaluator,
    best_agreement_aggregate,
)


def get_object_label(
    spin_poles: Optional[List[np.ndarray]],
    close_thresh_deg: float = 30.0,
) -> str:
    """
    Classify object based on its pole solutions.

    Args:
        spin_poles: List of pole vectors (xyz) from spin solutions, or None
        close_thresh_deg: Threshold for 'close' vs 'far' multi-pole classification

    Returns:
        'unique': single pole solution
        'multi_close': multiple poles within close_thresh_deg
        'multi_far': multiple poles farther apart
    """
    if spin_poles is None or len(spin_poles) <= 1:
        return 'unique'

    # Compute minimum angular separation between any pair of poles
    min_sep = float('inf')
    for i in range(len(spin_poles)):
        for j in range(i + 1, len(spin_poles)):
            sep = angular_sep_deg(spin_poles[i], spin_poles[j])
            min_sep = min(min_sep, sep)

    if min_sep <= close_thresh_deg:
        return 'multi_close'
    return 'multi_far'


def antipode_aware_angular_error(
    pred: np.ndarray,
    target: np.ndarray,
) -> float:
    """
    Compute antipode-aware angular error between prediction and target.

    Args:
        pred: Predicted pole, shape (3,)
        target: Target pole, shape (3,)

    Returns:
        Angular error in degrees, range [0, 90]
    """
    return angular_sep_deg(pred, target)


def compute_set_error(
    pred: np.ndarray,
    solutions_xyz: np.ndarray,
    topk: Optional[int] = None,
) -> float:
    """
    Compute minimum angular error over all valid pole solutions.

    Args:
        pred: Predicted pole, shape (3,)
        solutions_xyz: Valid pole solutions, shape (K, 3)
        topk: Optional cap on number of solutions to consider

    Returns:
        Minimum angular error in degrees
    """
    if solutions_xyz is None or len(solutions_xyz) == 0:
        # No solutions - should not happen, but return 90 as max error
        return 90.0

    if topk is not None and topk > 0:
        solutions_xyz = solutions_xyz[:topk]

    errors = [angular_sep_deg(pred, sol) for sol in solutions_xyz]
    return float(np.min(errors))


def _run_frozen_window_evaluation(
    evaluator: FrozenWindowEvaluator,
    model: torch.nn.Module,
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Run frozen window evaluation with best-agreement aggregation.

    This replicates the exact pipeline from phase28b_train_phase15_baseline.py.

    Args:
        evaluator: FrozenWindowEvaluator instance
        model: Trained model
        device: Device string
        batch_size: Batch size for inference

    Returns:
        Best-agreement predictions, shape (n_objects, 3)
    """
    model.eval()
    device_obj = torch.device(device)

    windows = evaluator.get_all_windows()
    n_objects = len(evaluator.objects)
    n_windows = evaluator.n_windows

    all_predictions = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            end = min(start + batch_size, len(windows))
            batch_windows = windows[start:end]

            tokens_batch = torch.stack(
                [torch.from_numpy(w[1]) for w in batch_windows]
            ).to(device_obj)
            masks_batch = torch.stack(
                [torch.from_numpy(w[2]) for w in batch_windows]
            ).to(device_obj)

            pred_poles = model(tokens_batch, masks_batch)
            all_predictions.append(pred_poles.cpu().numpy())

    all_predictions = np.concatenate(all_predictions, axis=0)

    # Best-agreement aggregation (same as Phase 28B)
    best_preds = best_agreement_aggregate(all_predictions, n_objects, n_windows)

    return best_preds


def eval_with_ambiguity(
    evaluator: FrozenWindowEvaluator,
    model: torch.nn.Module,
    object_ids: List[str],
    spin_db: Dict,
    device: str,
    seed: int,
    close_thresh_deg: float = 15.0,
    topk_valid_solutions: Optional[int] = None,
    batch_size: int = 64,
) -> Dict:
    """
    Evaluate model with ambiguity-aware metrics.

    Uses frozen windows + best_agreement aggregation (same as Phase 28B),
    then computes both canonical and set metrics, stratified by ambiguity class.

    Args:
        evaluator: FrozenWindowEvaluator for the validation fold
        model: Trained torch model
        object_ids: List of object IDs for validation objects
        spin_db: Spin solution database from spin_solutions.py
        device: Device string ("cuda" or "cpu")
        seed: Random seed (for reproducibility, though evaluation is deterministic)
        close_thresh_deg: Threshold for multi_close vs multi_far
        topk_valid_solutions: Optional cap on solutions for set error
        batch_size: Batch size for inference

    Returns:
        Dict with:
        - overall: {canonical_median, canonical_acc25, set_median, set_acc25, n}
        - by_label: {label -> same metrics + n}
        - per_object: list of per-object records
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Build ambiguity index
    ambiguity_index = build_ambiguity_index(spin_db, object_ids, close_thresh_deg)

    # Run frozen window evaluation
    best_preds = _run_frozen_window_evaluation(evaluator, model, device, batch_size)

    # Compute per-object metrics
    per_object = []
    n_objects = len(evaluator.objects)

    for obj_idx in range(n_objects):
        obj = evaluator.objects[obj_idx]
        obj_id = obj.get('object_id', object_ids[obj_idx] if obj_idx < len(object_ids) else f'obj_{obj_idx}')

        pred = best_preds[obj_idx]
        target = obj['pole']

        # Canonical error (vs single label)
        canonical_err = antipode_aware_angular_error(pred, target)

        # Get solutions for set error
        asteroid_id = extract_asteroid_id(obj_id)
        if asteroid_id in spin_db:
            sol = spin_db[asteroid_id]
            solutions_xyz = sol.poles_ecl_xyz if len(sol.poles_ecl_xyz) > 0 else None
        else:
            solutions_xyz = None

        # Set error (vs nearest valid pole)
        if solutions_xyz is not None and len(solutions_xyz) > 0:
            set_err = compute_set_error(pred, solutions_xyz, topk_valid_solutions)
        else:
            # Fall back to canonical pole as singleton
            set_err = canonical_err

        # Get ambiguity info
        if obj_id in ambiguity_index:
            info = ambiguity_index[obj_id]
            n_solutions = info.n_solutions
            label = info.label
            min_sep = info.min_sep_deg
        else:
            n_solutions = 1
            label = "unique"
            min_sep = None

        per_object.append({
            'object_id': obj_id,
            'canonical_err': canonical_err,
            'set_err': set_err,
            'n_solutions': n_solutions,
            'label': label,
            'min_sep_deg': min_sep,
        })

    # Aggregate metrics
    def compute_metrics(records: List[Dict]) -> Dict:
        if not records:
            return {
                'canonical_median': None,
                'canonical_acc25': None,
                'set_median': None,
                'set_acc25': None,
                'n': 0,
            }

        canonical_errs = [r['canonical_err'] for r in records]
        set_errs = [r['set_err'] for r in records]

        return {
            'canonical_median': float(np.median(canonical_errs)),
            'canonical_acc25': float(np.mean([e <= 25.0 for e in canonical_errs])),
            'set_median': float(np.median(set_errs)),
            'set_acc25': float(np.mean([e <= 25.0 for e in set_errs])),
            'n': len(records),
        }

    # Overall metrics
    overall = compute_metrics(per_object)

    # By-label metrics
    by_label = {}
    for label in ['unique', 'multi_close', 'multi_far']:
        label_records = [r for r in per_object if r['label'] == label]
        by_label[label] = compute_metrics(label_records)

    return {
        'overall': overall,
        'by_label': by_label,
        'per_object': per_object,
    }


def format_metrics_table(metrics: Dict, title: str = "Metrics") -> str:
    """
    Format metrics dict as markdown table.

    Args:
        metrics: Metrics dict from eval_with_ambiguity
        title: Table title

    Returns:
        Markdown-formatted string
    """
    lines = [
        f"## {title}",
        "",
        "### Overall",
        "",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    overall = metrics['overall']
    if overall['n'] > 0:
        lines.extend([
            f"| N | {overall['n']} |",
            f"| Canonical Median | {overall['canonical_median']:.2f}° |",
            f"| Canonical Acc@25 | {100*overall['canonical_acc25']:.1f}% |",
            f"| Set Median | {overall['set_median']:.2f}° |",
            f"| Set Acc@25 | {100*overall['set_acc25']:.1f}% |",
            f"| Gap (Can - Set) | {overall['canonical_median'] - overall['set_median']:.2f}° |",
        ])
    else:
        lines.append("| N | 0 |")

    lines.extend([
        "",
        "### By Ambiguity Label",
        "",
        "| Label | N | Can Median | Can Acc@25 | Set Median | Set Acc@25 |",
        "|-------|---|------------|------------|------------|------------|",
    ])

    for label in ['unique', 'multi_close', 'multi_far']:
        m = metrics['by_label'].get(label, {'n': 0})
        if m['n'] > 0:
            lines.append(
                f"| {label} | {m['n']} | {m['canonical_median']:.2f}° | "
                f"{100*m['canonical_acc25']:.1f}% | {m['set_median']:.2f}° | "
                f"{100*m['set_acc25']:.1f}% |"
            )
        else:
            lines.append(f"| {label} | 0 | - | - | - | - |")

    return '\n'.join(lines)


# =============================================================================
# Phase 29F: Multi-Hypothesis Evaluation Support
# =============================================================================

def eval_multi_hypothesis_with_selection(
    evaluator: FrozenWindowEvaluator,
    model,
    object_ids: List[str],
    spin_db: Dict,
    device: str,
    seed: int,
    close_thresh_deg: float = 15.0,
    topk_valid_solutions: Optional[int] = None,
    batch_size: int = 64,
    selector_mode: str = "agreement",
) -> Dict:
    """
    Evaluate multi-hypothesis model with all selection modes.

    Computes metrics for three selection strategies:
    1. oracle: Min error across all hypotheses (upper bound, not deployable)
    2. selected: Use hypothesis chosen by agreement/dispersion selector (deployable)
    3. naive0: Always use hypothesis 0 (baseline)

    Args:
        evaluator: FrozenWindowEvaluator for the validation fold
        model: Trained multi-hypothesis torch model
        object_ids: List of object IDs for validation objects
        spin_db: Spin solution database from spin_solutions.py
        device: Device string ("cuda" or "cpu")
        seed: Random seed for reproducibility
        close_thresh_deg: Threshold for multi_close vs multi_far
        topk_valid_solutions: Optional cap on solutions for set error
        batch_size: Batch size for inference
        selector_mode: Mode for hypothesis selection ("agreement" or "dispersion")

    Returns:
        Dict with:
        - oracle: {overall, by_label} using oracle (min-error) selection
        - selected: {overall, by_label} using deployed selector
        - naive0: {overall, by_label} using hypothesis 0
        - per_object: list of per-object records with all error types
        - diagnostics: collapse/complementarity stats
    """
    from pole_synth.damit_real_multiepoch_dataset import predict_all_objects_multi_hypothesis
    from pole_synth.hypothesis_selector import select_hypothesis_agreement, get_selection_scores

    np.random.seed(seed)

    # Build ambiguity index
    ambiguity_index = build_ambiguity_index(spin_db, object_ids, close_thresh_deg)

    # Get multi-hypothesis predictions
    pred_data = predict_all_objects_multi_hypothesis(evaluator, model, device, batch_size)
    hyp_xyz = pred_data['hyp_xyz']  # (N, K, 3)
    window_preds_xyz = pred_data['window_preds_xyz']  # (N, W, K, 3)

    N, K, _ = hyp_xyz.shape
    n_objects = len(evaluator.objects)

    assert N == n_objects, f"Mismatch: {N} predictions vs {n_objects} objects"

    # Compute per-object metrics
    per_object = []

    for obj_idx in range(n_objects):
        obj = evaluator.objects[obj_idx]
        obj_id = obj.get('object_id', object_ids[obj_idx] if obj_idx < len(object_ids) else f'obj_{obj_idx}')

        obj_hyp = hyp_xyz[obj_idx]  # (K, 3)
        obj_window_preds = window_preds_xyz[obj_idx]  # (W, K, 3)
        target = obj['pole']

        # Get solutions for set error
        asteroid_id = extract_asteroid_id(obj_id)
        if asteroid_id in spin_db:
            sol = spin_db[asteroid_id]
            solutions_xyz = sol.poles_ecl_xyz if len(sol.poles_ecl_xyz) > 0 else None
        else:
            solutions_xyz = None

        # Get ambiguity info
        if obj_id in ambiguity_index:
            info = ambiguity_index[obj_id]
            n_solutions = info.n_solutions
            label = info.label
            min_sep = info.min_sep_deg
        else:
            n_solutions = 1
            label = "unique"
            min_sep = None

        # Compute errors for each hypothesis
        hyp_canonical_errs = []
        hyp_set_errs = []

        for k in range(K):
            pred_k = obj_hyp[k]

            # Canonical error
            can_err = antipode_aware_angular_error(pred_k, target)
            hyp_canonical_errs.append(can_err)

            # Set error
            if solutions_xyz is not None and len(solutions_xyz) > 0:
                set_err = compute_set_error(pred_k, solutions_xyz, topk_valid_solutions)
            else:
                set_err = can_err
            hyp_set_errs.append(set_err)

        # Oracle: min error across hypotheses
        oracle_can_err = min(hyp_canonical_errs)
        oracle_set_err = min(hyp_set_errs)
        oracle_idx = int(np.argmin(hyp_set_errs))

        # Naive0: always hypothesis 0
        naive0_can_err = hyp_canonical_errs[0]
        naive0_set_err = hyp_set_errs[0]

        # Selected: use agreement/dispersion selector
        selected_idx, scores = get_selection_scores(
            obj_window_preds, obj_hyp, mode=selector_mode
        )
        selected_can_err = hyp_canonical_errs[selected_idx]
        selected_set_err = hyp_set_errs[selected_idx]

        # Diagnostics
        if K >= 2:
            interhyp_angle = angular_sep_deg(obj_hyp[0], obj_hyp[1])
            hyp1_better = hyp_set_errs[1] < hyp_set_errs[0]
            hyp1_better_by = hyp_set_errs[0] - hyp_set_errs[1]
        else:
            interhyp_angle = 0.0
            hyp1_better = False
            hyp1_better_by = 0.0

        per_object.append({
            'object_id': obj_id,
            'label': label,
            'n_solutions': n_solutions,
            'min_sep_deg': min_sep,
            # Oracle
            'can_err_oracle': oracle_can_err,
            'set_err_oracle': oracle_set_err,
            'oracle_idx': oracle_idx,
            # Selected
            'can_err_selected': selected_can_err,
            'set_err_selected': selected_set_err,
            'selected_idx': selected_idx,
            'selector_scores': scores.tolist(),
            # Naive0
            'can_err_naive0': naive0_can_err,
            'set_err_naive0': naive0_set_err,
            # Diagnostics
            'interhyp_angle': interhyp_angle,
            'hyp1_better': hyp1_better,
            'hyp1_better_by': hyp1_better_by,
            # All hypothesis errors
            'hyp_canonical_errs': hyp_canonical_errs,
            'hyp_set_errs': hyp_set_errs,
        })

    # Compute aggregate metrics for each selection mode
    def compute_metrics_for_mode(records: List[Dict], mode: str) -> Dict:
        if not records:
            return {
                'canonical_median': None,
                'canonical_acc25': None,
                'set_median': None,
                'set_acc25': None,
                'n': 0,
            }

        can_key = f'can_err_{mode}'
        set_key = f'set_err_{mode}'

        canonical_errs = [r[can_key] for r in records]
        set_errs = [r[set_key] for r in records]

        return {
            'canonical_median': float(np.median(canonical_errs)),
            'canonical_acc25': float(np.mean([e <= 25.0 for e in canonical_errs])),
            'set_median': float(np.median(set_errs)),
            'set_acc25': float(np.mean([e <= 25.0 for e in set_errs])),
            'n': len(records),
        }

    def compute_all_modes(records: List[Dict]) -> Dict:
        result = {}
        for mode in ['oracle', 'selected', 'naive0']:
            overall = compute_metrics_for_mode(records, mode)

            by_label = {}
            for label in ['unique', 'multi_close', 'multi_far']:
                label_records = [r for r in records if r['label'] == label]
                by_label[label] = compute_metrics_for_mode(label_records, mode)

            result[mode] = {
                'overall': overall,
                'by_label': by_label,
            }
        return result

    modes_metrics = compute_all_modes(per_object)

    # Compute diagnostics
    diagnostics = compute_multi_hypothesis_diagnostics(per_object, K)

    return {
        'oracle': modes_metrics['oracle'],
        'selected': modes_metrics['selected'],
        'naive0': modes_metrics['naive0'],
        'per_object': per_object,
        'diagnostics': diagnostics,
        'selector_mode': selector_mode,
        'n_hypotheses': K,
    }


def compute_multi_hypothesis_diagnostics(
    per_object: List[Dict],
    K: int,
    collapse_threshold: float = 5.0,
    complementarity_threshold: float = 5.0,
) -> Dict:
    """
    Compute diagnostic statistics for multi-hypothesis predictions.

    Args:
        per_object: List of per-object records from eval_multi_hypothesis_with_selection
        K: Number of hypotheses
        collapse_threshold: Angle below which hypotheses are considered collapsed (degrees)
        complementarity_threshold: Improvement by which hyp1 must beat hyp0 (degrees)

    Returns:
        Dict with:
        - collapse_rate: Fraction of objects with collapsed hypotheses
        - mean_interhyp_angle: Mean inter-hypothesis angle
        - mean_interhyp_angle_by_label: Per-label means
        - complementarity_rate: Fraction where hyp1 beats hyp0 by threshold
        - selector_accuracy: Fraction where selector picks oracle hypothesis
    """
    if K < 2:
        return {
            'collapse_rate': 0.0,
            'mean_interhyp_angle': 0.0,
            'mean_interhyp_angle_by_label': {},
            'complementarity_rate': 0.0,
            'selector_accuracy': 1.0,
        }

    n_collapsed = 0
    n_complementary = 0
    n_selector_correct = 0
    interhyp_angles = []
    interhyp_by_label = {label: [] for label in ['unique', 'multi_close', 'multi_far']}

    for r in per_object:
        interhyp_angle = r.get('interhyp_angle', 0.0)
        interhyp_angles.append(interhyp_angle)

        label = r.get('label', 'unique')
        if label in interhyp_by_label:
            interhyp_by_label[label].append(interhyp_angle)

        if interhyp_angle < collapse_threshold:
            n_collapsed += 1

        if r.get('hyp1_better_by', 0.0) >= complementarity_threshold:
            n_complementary += 1

        if r.get('selected_idx', 0) == r.get('oracle_idx', 0):
            n_selector_correct += 1

    N = len(per_object)

    mean_by_label = {}
    for label, angles in interhyp_by_label.items():
        if angles:
            mean_by_label[label] = float(np.mean(angles))
        else:
            mean_by_label[label] = None

    return {
        'collapse_rate': n_collapsed / max(N, 1),
        'mean_interhyp_angle': float(np.mean(interhyp_angles)) if interhyp_angles else 0.0,
        'mean_interhyp_angle_by_label': mean_by_label,
        'complementarity_rate': n_complementary / max(N, 1),
        'selector_accuracy': n_selector_correct / max(N, 1),
        'n_objects': N,
        'n_collapsed': n_collapsed,
        'n_complementary': n_complementary,
        'n_selector_correct': n_selector_correct,
    }


def format_multi_hypothesis_report(
    metrics: Dict,
    title: str = "Phase 29F Multi-Hypothesis Evaluation",
) -> str:
    """
    Format multi-hypothesis metrics as markdown report.

    Args:
        metrics: Metrics dict from eval_multi_hypothesis_with_selection
        title: Report title

    Returns:
        Markdown-formatted string
    """
    lines = [
        f"# {title}",
        "",
        f"**Number of Hypotheses (K):** {metrics.get('n_hypotheses', 1)}",
        f"**Selector Mode:** {metrics.get('selector_mode', 'agreement')}",
        "",
    ]

    # Overall comparison table
    lines.extend([
        "## Overall Metrics by Selection Mode",
        "",
        "| Mode | Can Median | Can Acc@25 | Set Median | Set Acc@25 |",
        "|------|------------|------------|------------|------------|",
    ])

    for mode in ['oracle', 'selected', 'naive0']:
        m = metrics.get(mode, {}).get('overall', {})
        if m.get('n', 0) > 0:
            lines.append(
                f"| {mode} | {m['canonical_median']:.2f}° | "
                f"{100*m['canonical_acc25']:.1f}% | {m['set_median']:.2f}° | "
                f"{100*m['set_acc25']:.1f}% |"
            )
        else:
            lines.append(f"| {mode} | - | - | - | - |")

    # Gap analysis
    lines.extend(["", "### Selection Gap (vs Oracle)"])
    oracle_set = metrics.get('oracle', {}).get('overall', {}).get('set_median', 0)
    selected_set = metrics.get('selected', {}).get('overall', {}).get('set_median', 0)
    naive0_set = metrics.get('naive0', {}).get('overall', {}).get('set_median', 0)

    if oracle_set is not None and selected_set is not None:
        lines.append(f"- Selected vs Oracle gap: {selected_set - oracle_set:.2f}°")
    if oracle_set is not None and naive0_set is not None:
        lines.append(f"- Naive0 vs Oracle gap: {naive0_set - oracle_set:.2f}°")

    # Per-label breakdown for selected mode
    lines.extend([
        "",
        "## Selected Mode (Deployable) by Ambiguity Label",
        "",
        "| Label | N | Can Median | Can Acc@25 | Set Median | Set Acc@25 |",
        "|-------|---|------------|------------|------------|------------|",
    ])

    selected_by_label = metrics.get('selected', {}).get('by_label', {})
    for label in ['unique', 'multi_close', 'multi_far']:
        m = selected_by_label.get(label, {'n': 0})
        if m.get('n', 0) > 0:
            lines.append(
                f"| {label} | {m['n']} | {m['canonical_median']:.2f}° | "
                f"{100*m['canonical_acc25']:.1f}% | {m['set_median']:.2f}° | "
                f"{100*m['set_acc25']:.1f}% |"
            )
        else:
            lines.append(f"| {label} | 0 | - | - | - | - |")

    # Diagnostics
    diag = metrics.get('diagnostics', {})
    lines.extend([
        "",
        "## Diagnostics",
        "",
        f"- **Collapse Rate:** {100*diag.get('collapse_rate', 0):.1f}% "
        f"(hypotheses within 5° of each other)",
        f"- **Mean Inter-Hypothesis Angle:** {diag.get('mean_interhyp_angle', 0):.1f}°",
        f"- **Complementarity Rate:** {100*diag.get('complementarity_rate', 0):.1f}% "
        f"(hyp1 beats hyp0 by ≥5° on set error)",
        f"- **Selector Accuracy:** {100*diag.get('selector_accuracy', 0):.1f}% "
        f"(selector picks oracle hypothesis)",
    ])

    # Per-label inter-hypothesis angles
    interhyp_by_label = diag.get('mean_interhyp_angle_by_label', {})
    if interhyp_by_label:
        lines.extend(["", "### Inter-Hypothesis Angle by Label"])
        for label in ['unique', 'multi_close', 'multi_far']:
            val = interhyp_by_label.get(label)
            if val is not None:
                lines.append(f"- {label}: {val:.1f}°")

    return '\n'.join(lines)


# =============================================================================
# Phase 29H: Learned Selector Evaluation
# =============================================================================

def eval_multi_hypothesis_with_learned_selector(
    evaluator,
    model,
    objects: List[Dict],
    spin_db: Optional[Dict] = None,
    device: str = "cuda",
    batch_size: int = 64,
    selector_checkpoint: Optional[Path] = None,
    selector_model: Optional[Any] = None,
    close_thresh_deg: float = 15.0,
) -> Dict[str, Any]:
    """
    Evaluate K=2 model with both learned and heuristic selectors.

    Computes metrics for:
    - Oracle@K (best possible)
    - Learned selector
    - Agreement heuristic (baseline)
    - Dispersion heuristic
    - Naive0 (always pick hyp 0)

    Args:
        evaluator: FrozenWindowEvaluator
        model: MultiHypothesisPoleRegressor
        objects: List of object dicts with ground truth
        spin_db: Spin solution database
        device: Device string
        batch_size: Batch size
        selector_checkpoint: Path to learned selector checkpoint
        selector_model: Pre-loaded selector model
        close_thresh_deg: Threshold for multi_close vs multi_far

    Returns:
        Dict with per-selector metrics and per-object details
    """
    from pole_synth.inference import predict_pole_selected
    from pole_synth.selector_model import load_selector

    # Pre-load learned selector
    loaded_selector = None
    if selector_checkpoint is not None and selector_model is None:
        loaded_selector = load_selector(Path(selector_checkpoint), device=device)

    per_object = []

    for obj_idx in range(len(objects)):
        obj = objects[obj_idx]
        object_id = obj.get('object_id', f'obj_{obj_idx}')

        # Get ground truth
        target_xyz = obj.get('pole') or obj.get('pole_ecl_xyz')
        if target_xyz is None:
            continue
        target_xyz = np.array(target_xyz)

        # Get spin solutions for set error
        asteroid_id = object_id.split('_')[1] if '_' in object_id else object_id
        spin_poles = None
        if spin_db is not None:
            spin_sol = spin_db.get(asteroid_id)
            if spin_sol is not None and hasattr(spin_sol, 'poles_ecl_xyz'):
                spin_poles = spin_sol.poles_ecl_xyz

        # Label
        label = get_object_label(spin_poles, close_thresh_deg) if spin_poles is not None else 'unique'

        # Predict with different selectors
        try:
            result_agreement = predict_pole_selected(
                model, evaluator, obj_idx, device, batch_size,
                selector_mode="agreement",
            )
            result_dispersion = predict_pole_selected(
                model, evaluator, obj_idx, device, batch_size,
                selector_mode="dispersion",
            )

            hyp_xyz = result_agreement['hyp_xyz']
            K = len(hyp_xyz)

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

            # Selector results
            record = {
                'object_id': object_id,
                'label': label,
                'n_solutions': len(spin_poles) if spin_poles is not None else 1,
                'hyp_xyz': hyp_xyz,
                # Oracle
                'can_err_oracle': can_err_oracle,
                'set_err_oracle': set_err_oracle,
                'oracle_idx': oracle_idx,
                # Agreement
                'can_err_agreement': can_errors[result_agreement['selected_idx']],
                'set_err_agreement': set_errors[result_agreement['selected_idx']],
                'agreement_idx': result_agreement['selected_idx'],
                # Dispersion
                'can_err_dispersion': can_errors[result_dispersion['selected_idx']],
                'set_err_dispersion': set_errors[result_dispersion['selected_idx']],
                'dispersion_idx': result_dispersion['selected_idx'],
                # Naive0
                'can_err_naive0': can_errors[0],
                'set_err_naive0': set_errors[0],
                # Inter-hypothesis
                'interhyp_angle': angular_sep_deg(hyp_xyz[0], hyp_xyz[1]) if K == 2 else 0,
            }

            # Learned selector (if available)
            if selector_checkpoint is not None or selector_model is not None:
                result_learned = predict_pole_selected(
                    model, evaluator, obj_idx, device, batch_size,
                    selector_mode="learned",
                    selector_model=selector_model or loaded_selector,
                )
                record['can_err_learned'] = can_errors[result_learned['selected_idx']]
                record['set_err_learned'] = set_errors[result_learned['selected_idx']]
                record['learned_idx'] = result_learned['selected_idx']
                record['learned_confidence'] = result_learned.get('confidence', 1.0)
                record['learned_probs'] = result_learned.get('selector_scores')

            per_object.append(record)

        except Exception as e:
            continue

    # Aggregate metrics
    metrics = _aggregate_learned_selector_metrics(per_object)
    metrics['per_object'] = per_object

    return metrics


def _aggregate_learned_selector_metrics(per_object: List[Dict]) -> Dict[str, Any]:
    """Aggregate per-object results into summary metrics."""
    if not per_object:
        return {}

    has_learned = 'can_err_learned' in per_object[0]

    # Collect errors
    can_oracle = [r['can_err_oracle'] for r in per_object]
    set_oracle = [r['set_err_oracle'] for r in per_object]
    can_agreement = [r['can_err_agreement'] for r in per_object]
    set_agreement = [r['set_err_agreement'] for r in per_object]
    can_dispersion = [r['can_err_dispersion'] for r in per_object]
    set_dispersion = [r['set_err_dispersion'] for r in per_object]
    can_naive0 = [r['can_err_naive0'] for r in per_object]
    set_naive0 = [r['set_err_naive0'] for r in per_object]

    metrics = {
        'n_objects': len(per_object),
        # Oracle
        'can_err_oracle_mean': float(np.mean(can_oracle)),
        'can_err_oracle_median': float(np.median(can_oracle)),
        'set_err_oracle_mean': float(np.mean(set_oracle)),
        'set_err_oracle_median': float(np.median(set_oracle)),
        # Agreement
        'can_err_agreement_mean': float(np.mean(can_agreement)),
        'can_err_agreement_median': float(np.median(can_agreement)),
        'set_err_agreement_mean': float(np.mean(set_agreement)),
        'set_err_agreement_median': float(np.median(set_agreement)),
        # Dispersion
        'can_err_dispersion_mean': float(np.mean(can_dispersion)),
        'can_err_dispersion_median': float(np.median(can_dispersion)),
        'set_err_dispersion_mean': float(np.mean(set_dispersion)),
        'set_err_dispersion_median': float(np.median(set_dispersion)),
        # Naive0
        'can_err_naive0_mean': float(np.mean(can_naive0)),
        'can_err_naive0_median': float(np.median(can_naive0)),
        'set_err_naive0_mean': float(np.mean(set_naive0)),
        'set_err_naive0_median': float(np.median(set_naive0)),
    }

    # Selector accuracy
    oracle_indices = [r['oracle_idx'] for r in per_object]
    agreement_indices = [r['agreement_idx'] for r in per_object]
    dispersion_indices = [r['dispersion_idx'] for r in per_object]

    metrics['agreement_accuracy'] = float(np.mean([
        a == o for a, o in zip(agreement_indices, oracle_indices)
    ]))
    metrics['dispersion_accuracy'] = float(np.mean([
        d == o for d, o in zip(dispersion_indices, oracle_indices)
    ]))

    # Learned selector
    if has_learned:
        can_learned = [r['can_err_learned'] for r in per_object]
        set_learned = [r['set_err_learned'] for r in per_object]
        learned_indices = [r['learned_idx'] for r in per_object]
        confidences = [r['learned_confidence'] for r in per_object]

        metrics['can_err_learned_mean'] = float(np.mean(can_learned))
        metrics['can_err_learned_median'] = float(np.median(can_learned))
        metrics['set_err_learned_mean'] = float(np.mean(set_learned))
        metrics['set_err_learned_median'] = float(np.median(set_learned))

        metrics['learned_accuracy'] = float(np.mean([
            l == o for l, o in zip(learned_indices, oracle_indices)
        ]))
        metrics['learned_confidence_mean'] = float(np.mean(confidences))

        # Gaps
        metrics['oracle_gap_agreement'] = metrics['set_err_agreement_mean'] - metrics['set_err_oracle_mean']
        metrics['oracle_gap_learned'] = metrics['set_err_learned_mean'] - metrics['set_err_oracle_mean']
        metrics['learned_vs_agreement'] = metrics['set_err_agreement_mean'] - metrics['set_err_learned_mean']

    # Per-label breakdown
    for label in ['unique', 'multi_close', 'multi_far']:
        label_objs = [r for r in per_object if r['label'] == label]
        if label_objs:
            metrics[f'{label}_count'] = len(label_objs)
            metrics[f'{label}_can_err_oracle_mean'] = float(np.mean([r['can_err_oracle'] for r in label_objs]))
            metrics[f'{label}_set_err_oracle_mean'] = float(np.mean([r['set_err_oracle'] for r in label_objs]))
            metrics[f'{label}_set_err_agreement_mean'] = float(np.mean([r['set_err_agreement'] for r in label_objs]))

            if has_learned:
                metrics[f'{label}_set_err_learned_mean'] = float(np.mean([r['set_err_learned'] for r in label_objs]))

    return metrics


def compute_selector_calibration(
    per_object: List[Dict],
    n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Compute calibration metrics for learned selector.

    Bins objects by selector confidence and checks if higher confidence
    correlates with lower oracle gap (selector being correct more often).

    Args:
        per_object: List of per-object records with learned_confidence
        n_bins: Number of confidence bins

    Returns:
        Dict with calibration statistics per bin
    """
    if not per_object or 'learned_confidence' not in per_object[0]:
        return {}

    # Filter objects with learned confidence
    objs = [r for r in per_object if 'learned_confidence' in r]
    if len(objs) < n_bins:
        return {}

    confidences = np.array([r['learned_confidence'] for r in objs])
    gaps = np.array([r['set_err_learned'] - r['set_err_oracle'] for r in objs])
    correct = np.array([r['learned_idx'] == r['oracle_idx'] for r in objs])

    # Create bins by confidence percentile
    bin_edges = np.percentile(confidences, np.linspace(0, 100, n_bins + 1))
    bin_edges[-1] += 0.01  # Include max

    bins = []
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bins.append({
                'bin_idx': i,
                'conf_low': float(bin_edges[i]),
                'conf_high': float(bin_edges[i + 1]),
                'conf_mean': float(confidences[mask].mean()),
                'n_objects': int(mask.sum()),
                'accuracy': float(correct[mask].mean()),
                'gap_mean': float(gaps[mask].mean()),
                'gap_median': float(np.median(gaps[mask])),
            })

    # Check monotonicity: higher confidence should correlate with lower gap
    if len(bins) >= 2:
        gap_means = [b['gap_mean'] for b in bins]
        monotonic = all(gap_means[i] >= gap_means[i + 1] for i in range(len(gap_means) - 1))
    else:
        monotonic = True

    return {
        'n_bins': len(bins),
        'bins': bins,
        'is_calibrated': monotonic,
        'overall_accuracy': float(correct.mean()),
        'overall_gap_mean': float(gaps.mean()),
    }


def format_learned_selector_report(
    metrics: Dict[str, Any],
    calibration: Optional[Dict[str, Any]] = None,
    title: str = "Phase 29H: Learned Selector Evaluation",
) -> str:
    """
    Format markdown report for learned selector evaluation.

    Args:
        metrics: Output from eval_multi_hypothesis_with_learned_selector
        calibration: Output from compute_selector_calibration (optional)
        title: Report title

    Returns:
        Markdown formatted report string
    """
    lines = [
        f"# {title}",
        "",
        f"**Objects evaluated:** {metrics.get('n_objects', 0)}",
        "",
        "## Set Error Summary (degrees)",
        "",
        "| Selector | Mean | Median |",
        "|----------|------|--------|",
        f"| Oracle@K | {metrics.get('set_err_oracle_mean', 0):.2f} | {metrics.get('set_err_oracle_median', 0):.2f} |",
    ]

    if 'set_err_learned_mean' in metrics:
        lines.append(
            f"| Learned | {metrics.get('set_err_learned_mean', 0):.2f} | {metrics.get('set_err_learned_median', 0):.2f} |"
        )

    lines.extend([
        f"| Agreement | {metrics.get('set_err_agreement_mean', 0):.2f} | {metrics.get('set_err_agreement_median', 0):.2f} |",
        f"| Dispersion | {metrics.get('set_err_dispersion_mean', 0):.2f} | {metrics.get('set_err_dispersion_median', 0):.2f} |",
        f"| Naive0 | {metrics.get('set_err_naive0_mean', 0):.2f} | {metrics.get('set_err_naive0_median', 0):.2f} |",
        "",
        "## Selector Accuracy",
        "",
    ])

    if 'learned_accuracy' in metrics:
        lines.append(f"- **Learned:** {100*metrics.get('learned_accuracy', 0):.1f}%")

    lines.extend([
        f"- **Agreement:** {100*metrics.get('agreement_accuracy', 0):.1f}%",
        f"- **Dispersion:** {100*metrics.get('dispersion_accuracy', 0):.1f}%",
    ])

    # Oracle gap comparison
    if 'oracle_gap_learned' in metrics:
        lines.extend([
            "",
            "## Oracle Gap Analysis",
            "",
            f"- **Agreement Gap:** {metrics.get('oracle_gap_agreement', 0):.2f}° (Selected - Oracle)",
            f"- **Learned Gap:** {metrics.get('oracle_gap_learned', 0):.2f}° (Selected - Oracle)",
            f"- **Learned vs Agreement:** {metrics.get('learned_vs_agreement', 0):.2f}° improvement",
        ])

    # Per-label breakdown
    lines.extend(["", "## Per-Label Breakdown", ""])
    for label in ['unique', 'multi_close', 'multi_far']:
        count = metrics.get(f'{label}_count')
        if count:
            lines.append(f"### {label} (n={count})")
            lines.append(f"- Oracle Set Error: {metrics.get(f'{label}_set_err_oracle_mean', 0):.2f}°")
            lines.append(f"- Agreement Set Error: {metrics.get(f'{label}_set_err_agreement_mean', 0):.2f}°")
            if f'{label}_set_err_learned_mean' in metrics:
                lines.append(f"- Learned Set Error: {metrics.get(f'{label}_set_err_learned_mean', 0):.2f}°")
            lines.append("")

    # Calibration
    if calibration and calibration.get('bins'):
        lines.extend([
            "## Confidence Calibration",
            "",
            f"**Is Calibrated (monotonic):** {'Yes' if calibration.get('is_calibrated') else 'No'}",
            "",
            "| Confidence Range | N | Accuracy | Gap Mean |",
            "|------------------|---|----------|----------|",
        ])
        for b in calibration['bins']:
            lines.append(
                f"| [{b['conf_low']:.2f}, {b['conf_high']:.2f}) | {b['n_objects']} | "
                f"{100*b['accuracy']:.1f}% | {b['gap_mean']:.2f}° |"
            )

    return '\n'.join(lines)
