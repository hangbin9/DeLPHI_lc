"""
Phase 29I: Unified Report Builder.

Generates deterministic markdown reports and JSON artifacts for paper-ready results.
All outputs are reproducible given the same input data.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pole_synth.version import __version__, get_version_info


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def compute_mean_std(values: List[float]) -> Tuple[float, float]:
    """Compute mean and std from list of values."""
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    return float(np.mean(arr)), float(np.std(arr))


def format_mean_std(mean: float, std: float, precision: int = 2) -> str:
    """Format mean±std string."""
    return f"{mean:.{precision}f}±{std:.{precision}f}"


def aggregate_fold_results(
    fold_results: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate per-fold results into mean±std statistics.

    Args:
        fold_results: Dict mapping fold_id -> fold metrics dict
            Each fold dict should have structure like:
            {
                'oracle_k2': {'overall': {'set_err_mean': X, ...}, 'by_label': {...}},
                'selected_learned_k2': {...},
                'selected_agreement_k2': {...},
                'naive0_k2': {...},
                'calibration': {...},
                'diagnostics': {...},
            }

    Returns:
        Dict with aggregated mean±std for each metric
    """
    if not fold_results:
        return {}

    aggregates = {}

    # Metric groups to aggregate
    metric_groups = [
        'oracle_k2',
        'selected_learned_k2',
        'selected_agreement_k2',
        'naive0_k2',
    ]

    # Overall metrics to aggregate
    overall_metrics = [
        'set_err_mean',
        'set_err_median',
        'can_err_mean',
        'can_err_median',
    ]

    for group in metric_groups:
        aggregates[group] = {'overall': {}, 'by_label': {}}

        # Overall metrics
        for metric in overall_metrics:
            values = []
            for fold_id, fold_data in fold_results.items():
                if group in fold_data and 'overall' in fold_data[group]:
                    val = fold_data[group]['overall'].get(metric)
                    if val is not None:
                        values.append(val)
            if values:
                mean, std = compute_mean_std(values)
                aggregates[group]['overall'][f'{metric}_mean'] = mean
                aggregates[group]['overall'][f'{metric}_std'] = std

        # By-label metrics
        for label in ['unique', 'multi_close', 'multi_far']:
            aggregates[group]['by_label'][label] = {}
            for metric in overall_metrics:
                values = []
                for fold_id, fold_data in fold_results.items():
                    if group in fold_data and 'by_label' in fold_data[group]:
                        label_data = fold_data[group]['by_label'].get(label, {})
                        val = label_data.get(metric)
                        if val is not None:
                            values.append(val)
                if values:
                    mean, std = compute_mean_std(values)
                    aggregates[group]['by_label'][label][f'{metric}_mean'] = mean
                    aggregates[group]['by_label'][label][f'{metric}_std'] = std

    # Aggregate selector accuracy
    for selector in ['learned', 'agreement']:
        key = f'selected_{selector}_k2'
        if key in aggregates:
            acc_values = []
            for fold_id, fold_data in fold_results.items():
                if key in fold_data:
                    acc = fold_data[key].get('selector_accuracy')
                    if acc is not None:
                        acc_values.append(acc)
            if acc_values:
                mean, std = compute_mean_std(acc_values)
                aggregates[key]['selector_accuracy_mean'] = mean
                aggregates[key]['selector_accuracy_std'] = std

    # Aggregate oracle gap
    gap_values_learned = []
    gap_values_agreement = []
    for fold_id, fold_data in fold_results.items():
        oracle_mean = fold_data.get('oracle_k2', {}).get('overall', {}).get('set_err_mean')
        learned_mean = fold_data.get('selected_learned_k2', {}).get('overall', {}).get('set_err_mean')
        agreement_mean = fold_data.get('selected_agreement_k2', {}).get('overall', {}).get('set_err_mean')
        if oracle_mean is not None and learned_mean is not None:
            gap_values_learned.append(learned_mean - oracle_mean)
        if oracle_mean is not None and agreement_mean is not None:
            gap_values_agreement.append(agreement_mean - oracle_mean)

    if gap_values_learned:
        mean, std = compute_mean_std(gap_values_learned)
        aggregates['oracle_gap_learned'] = {'mean': mean, 'std': std}
    if gap_values_agreement:
        mean, std = compute_mean_std(gap_values_agreement)
        aggregates['oracle_gap_agreement'] = {'mean': mean, 'std': std}

    # Aggregate calibration
    calibration_bins = {}
    for fold_id, fold_data in fold_results.items():
        cal = fold_data.get('calibration', {})
        bins = cal.get('bins', [])
        for b in bins:
            bin_idx = b.get('bin_idx', 0)
            if bin_idx not in calibration_bins:
                calibration_bins[bin_idx] = {
                    'gap_means': [],
                    'accuracies': [],
                    'counts': [],
                }
            calibration_bins[bin_idx]['gap_means'].append(b.get('gap_mean', 0))
            calibration_bins[bin_idx]['accuracies'].append(b.get('accuracy', 0))
            calibration_bins[bin_idx]['counts'].append(b.get('n_objects', 0))

    aggregates['calibration'] = []
    for bin_idx in sorted(calibration_bins.keys()):
        data = calibration_bins[bin_idx]
        aggregates['calibration'].append({
            'bin_idx': bin_idx,
            'gap_mean': float(np.mean(data['gap_means'])) if data['gap_means'] else 0,
            'gap_std': float(np.std(data['gap_means'])) if data['gap_means'] else 0,
            'accuracy_mean': float(np.mean(data['accuracies'])) if data['accuracies'] else 0,
            'total_count': int(np.sum(data['counts'])) if data['counts'] else 0,
        })

    # Aggregate diagnostics
    diag_metrics = ['collapse_rate', 'complementarity_rate', 'mean_interhyp_angle']
    aggregates['diagnostics'] = {}
    for metric in diag_metrics:
        values = []
        for fold_id, fold_data in fold_results.items():
            diag = fold_data.get('diagnostics', {})
            val = diag.get(metric)
            if val is not None:
                values.append(val)
        if values:
            mean, std = compute_mean_std(values)
            aggregates['diagnostics'][f'{metric}_mean'] = mean
            aggregates['diagnostics'][f'{metric}_std'] = std

    return aggregates


def build_paper_results_json(
    fold_results: Dict[str, Dict[str, Any]],
    commands: List[str],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build complete paper results JSON structure.

    Args:
        fold_results: Per-fold results
        commands: CLI commands executed
        config: Run configuration

    Returns:
        Complete results dict ready for JSON serialization
    """
    aggregates = aggregate_fold_results(fold_results)

    results = {
        'version': __version__,
        'version_info': get_version_info(),
        'timestamp': datetime.now().isoformat(),
        'git_commit': get_git_commit_hash(),
        'config': config,
        'commands': commands,
        'n_folds': len(fold_results),
        'folds': fold_results,
        'aggregates': aggregates,
    }

    return results


def format_overall_metrics_table(
    aggregates: Dict[str, Any],
    title: str = "Overall Results (Mean±Std across folds)",
) -> str:
    """Format overall metrics as markdown table."""
    lines = [
        f"## {title}",
        "",
        "| Method | Set Error (Mean) | Set Error (Median) | Can Error (Mean) |",
        "|--------|-----------------|-------------------|-----------------|",
    ]

    methods = [
        ('Oracle@2', 'oracle_k2'),
        ('Selected@2 (Learned)', 'selected_learned_k2'),
        ('Selected@2 (Agreement)', 'selected_agreement_k2'),
        ('Naive0@2', 'naive0_k2'),
    ]

    for name, key in methods:
        data = aggregates.get(key, {}).get('overall', {})
        set_mean = format_mean_std(
            data.get('set_err_mean_mean', 0),
            data.get('set_err_mean_std', 0),
        )
        set_med = format_mean_std(
            data.get('set_err_median_mean', 0),
            data.get('set_err_median_std', 0),
        )
        can_mean = format_mean_std(
            data.get('can_err_mean_mean', 0),
            data.get('can_err_mean_std', 0),
        )
        lines.append(f"| {name} | {set_mean}° | {set_med}° | {can_mean}° |")

    return '\n'.join(lines)


def format_by_label_table(
    aggregates: Dict[str, Any],
    method_key: str,
    method_name: str,
) -> str:
    """Format by-label breakdown as markdown table."""
    lines = [
        f"### {method_name} by Label",
        "",
        "| Label | Set Error (Mean) | Set Error (Median) | N (approx) |",
        "|-------|-----------------|-------------------|------------|",
    ]

    data = aggregates.get(method_key, {}).get('by_label', {})

    for label in ['unique', 'multi_close', 'multi_far']:
        label_data = data.get(label, {})
        set_mean = format_mean_std(
            label_data.get('set_err_mean_mean', 0),
            label_data.get('set_err_mean_std', 0),
        )
        set_med = format_mean_std(
            label_data.get('set_err_median_mean', 0),
            label_data.get('set_err_median_std', 0),
        )
        lines.append(f"| {label} | {set_mean}° | {set_med}° | - |")

    return '\n'.join(lines)


def format_oracle_gap_table(aggregates: Dict[str, Any]) -> str:
    """Format oracle gap comparison table."""
    lines = [
        "## Oracle Gap Analysis",
        "",
        "Gap = Selected - Oracle (lower is better)",
        "",
        "| Selector | Mean Gap | Std |",
        "|----------|----------|-----|",
    ]

    for selector, key in [('Learned', 'oracle_gap_learned'), ('Agreement', 'oracle_gap_agreement')]:
        data = aggregates.get(key, {})
        mean = data.get('mean', 0)
        std = data.get('std', 0)
        lines.append(f"| {selector} | {mean:.2f}° | {std:.2f}° |")

    return '\n'.join(lines)


def format_calibration_table(aggregates: Dict[str, Any]) -> str:
    """Format calibration table."""
    cal_data = aggregates.get('calibration', [])
    if not cal_data:
        return "## Calibration\n\nNo calibration data available."

    lines = [
        "## Selector Calibration",
        "",
        "Confidence bins with mean oracle gap (higher confidence should → lower gap)",
        "",
        "| Bin | Gap Mean | Gap Std | Accuracy | Count |",
        "|-----|----------|---------|----------|-------|",
    ]

    for b in cal_data:
        lines.append(
            f"| {b['bin_idx']} | {b['gap_mean']:.2f}° | {b['gap_std']:.2f}° | "
            f"{100*b['accuracy_mean']:.1f}% | {b['total_count']} |"
        )

    # Check monotonicity
    if len(cal_data) >= 2:
        gaps = [b['gap_mean'] for b in cal_data]
        is_monotonic = all(gaps[i] >= gaps[i + 1] for i in range(len(gaps) - 1))
        lines.append("")
        lines.append(f"**Calibrated (monotonic):** {'Yes' if is_monotonic else 'No'}")

    return '\n'.join(lines)


def format_diagnostics_section(aggregates: Dict[str, Any]) -> str:
    """Format diagnostics section."""
    diag = aggregates.get('diagnostics', {})

    lines = [
        "## Diagnostics",
        "",
    ]

    collapse = diag.get('collapse_rate_mean', 0)
    collapse_std = diag.get('collapse_rate_std', 0)
    comp = diag.get('complementarity_rate_mean', 0)
    comp_std = diag.get('complementarity_rate_std', 0)
    interhyp = diag.get('mean_interhyp_angle_mean', 0)
    interhyp_std = diag.get('mean_interhyp_angle_std', 0)

    lines.extend([
        f"- **Collapse Rate:** {format_mean_std(100*collapse, 100*collapse_std)}% "
        f"(hypotheses within 5° of each other)",
        f"- **Complementarity Rate:** {format_mean_std(100*comp, 100*comp_std)}% "
        f"(hyp1 beats hyp0 by ≥5° on set error)",
        f"- **Mean Inter-Hypothesis Angle:** {format_mean_std(interhyp, interhyp_std)}°",
    ])

    # Selector accuracy
    for selector, key in [('Learned', 'selected_learned_k2'), ('Agreement', 'selected_agreement_k2')]:
        data = aggregates.get(key, {})
        acc_mean = data.get('selector_accuracy_mean')
        acc_std = data.get('selector_accuracy_std')
        if acc_mean is not None:
            lines.append(
                f"- **{selector} Selector Accuracy:** {format_mean_std(100*acc_mean, 100*acc_std)}%"
            )

    return '\n'.join(lines)


def format_commands_section(commands: List[str]) -> str:
    """Format executed commands section."""
    lines = [
        "## Commands Executed",
        "",
    ]

    for cmd in commands:
        lines.append("```bash")
        lines.append(cmd)
        lines.append("```")
        lines.append("")

    return '\n'.join(lines)


def build_paper_report(
    results: Dict[str, Any],
    title: str = "Pole Prediction Results (Phase 29I)",
) -> str:
    """
    Build complete paper-ready markdown report.

    Args:
        results: Output from build_paper_results_json
        title: Report title

    Returns:
        Complete markdown report string
    """
    sections = []

    # Header
    sections.append(f"# {title}")
    sections.append("")
    sections.append(f"**Version:** {results.get('version', 'unknown')}")
    sections.append(f"**Generated:** {results.get('timestamp', 'unknown')}")
    git_commit = results.get('git_commit')
    if git_commit:
        sections.append(f"**Git Commit:** {git_commit}")
    sections.append(f"**Folds:** {results.get('n_folds', 0)}")
    sections.append("")

    aggregates = results.get('aggregates', {})

    # Overall results
    sections.append(format_overall_metrics_table(aggregates))
    sections.append("")

    # By-label breakdowns
    sections.append("## By-Label Breakdown")
    sections.append("")
    for method_name, method_key in [
        ('Oracle@2', 'oracle_k2'),
        ('Selected@2 (Learned)', 'selected_learned_k2'),
        ('Selected@2 (Agreement)', 'selected_agreement_k2'),
    ]:
        sections.append(format_by_label_table(aggregates, method_key, method_name))
        sections.append("")

    # Oracle gap
    sections.append(format_oracle_gap_table(aggregates))
    sections.append("")

    # Calibration
    sections.append(format_calibration_table(aggregates))
    sections.append("")

    # Diagnostics
    sections.append(format_diagnostics_section(aggregates))
    sections.append("")

    # Commands
    commands = results.get('commands', [])
    if commands:
        sections.append(format_commands_section(commands))

    return '\n'.join(sections)


def save_paper_results(
    results: Dict[str, Any],
    json_path: Path,
    md_path: Path,
) -> None:
    """
    Save paper results to JSON and markdown files.

    Args:
        results: Output from build_paper_results_json
        json_path: Path for JSON output
        md_path: Path for markdown output
    """
    json_path = Path(json_path)
    md_path = Path(md_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save markdown
    report = build_paper_report(results)
    with open(md_path, 'w') as f:
        f.write(report)
