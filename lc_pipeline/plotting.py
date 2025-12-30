"""
Plotting utilities for period prediction diagnostics.

This module provides:
- plot_period_parity: true vs predicted period scatter plot
- plot_uncertainty_vs_error: error vs uncertainty scatter plot
- plot_error_histogram: histogram of prediction errors
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import alias_aware_relative_error


def plot_period_parity(
    truth_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    out_path: Optional[Union[str, Path]] = None,
    max_period_hours: float = 100.0,
    figsize: tuple = (8, 8),
    true_col: str = "period_hours",
    pred_col: str = "period_hours"
) -> plt.Figure:
    """
    Create a period parity plot (true vs predicted).

    Includes:
    - y=x perfect prediction line
    - y=0.5x and y=2x alias lines
    - Points colored by error magnitude

    Args:
        truth_df: DataFrame with object_id, period_hours (true).
        preds_df: DataFrame with object_id, period_hours (predicted).
        out_path: Optional path to save figure.
        max_period_hours: Maximum period for axis limits.
        figsize: Figure size tuple.
        true_col: Column name for true period.
        pred_col: Column name for predicted period.

    Returns:
        Matplotlib Figure object.
    """
    # Merge on object_id
    merged = preds_df.merge(
        truth_df[["object_id", true_col]],
        on="object_id",
        suffixes=("_pred", "_true")
    )

    if len(merged) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No matching data", ha='center', va='center',
                transform=ax.transAxes)
        return fig

    # Extract values
    if f"{true_col}_true" in merged.columns:
        true_vals = merged[f"{true_col}_true"].values
    else:
        true_vals = merged[true_col].values

    if pred_col in merged.columns:
        pred_vals = merged[pred_col].values
    elif f"{pred_col}_pred" in merged.columns:
        pred_vals = merged[f"{pred_col}_pred"].values
    else:
        pred_vals = merged[pred_col].values

    # Filter to finite values within range
    mask = (np.isfinite(true_vals) & np.isfinite(pred_vals) &
            (true_vals > 0) & (pred_vals > 0) &
            (true_vals <= max_period_hours) & (pred_vals <= max_period_hours))

    true_plot = true_vals[mask]
    pred_plot = pred_vals[mask]

    # Compute alias-aware errors for coloring
    errors = np.array([
        alias_aware_relative_error(p, t)
        for p, t in zip(pred_plot, true_plot)
    ])

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Reference lines
    x_ref = np.linspace(0.1, max_period_hours, 100)

    ax.plot(x_ref, x_ref, 'k-', lw=2, label='y = x (perfect)', zorder=1)
    ax.plot(x_ref, 0.5 * x_ref, 'k--', lw=1, alpha=0.5,
            label='y = 0.5x (half-period alias)', zorder=1)
    ax.plot(x_ref, 2.0 * x_ref, 'k:', lw=1, alpha=0.5,
            label='y = 2x (double-period alias)', zorder=1)

    # Scatter plot colored by error
    scatter = ax.scatter(
        true_plot, pred_plot,
        c=errors, cmap='RdYlGn_r',
        vmin=0, vmax=0.5,
        alpha=0.7, s=30, edgecolors='none',
        zorder=2
    )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Alias-aware relative error')

    # Labels and formatting
    ax.set_xlabel('True Period (hours)', fontsize=12)
    ax.set_ylabel('Predicted Period (hours)', fontsize=12)
    ax.set_title('Period Parity Plot', fontsize=14)

    ax.set_xlim(0, max_period_hours)
    ax.set_ylim(0, max_period_hours)
    ax.set_aspect('equal')

    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Stats annotation
    n_total = len(true_plot)
    n_good = np.sum(errors <= 0.05)
    pct_good = 100 * n_good / n_total if n_total > 0 else 0

    stats_text = f"N = {n_total}\nAcc@5% = {pct_good:.1f}%"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')

    return fig


def plot_uncertainty_vs_error(
    truth_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    out_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 6),
    true_col: str = "period_hours",
    pred_col: str = "period_hours",
    sigma_col: str = "sigma_eff_hours"
) -> plt.Figure:
    """
    Plot prediction error vs uncertainty estimate.

    Good uncertainty estimates should correlate with actual error.

    Args:
        truth_df: DataFrame with object_id, period_hours (true).
        preds_df: DataFrame with object_id, period_hours, sigma_eff_hours.
        out_path: Optional path to save figure.
        figsize: Figure size tuple.
        true_col: Column name for true period.
        pred_col: Column name for predicted period.
        sigma_col: Column name for uncertainty estimate.

    Returns:
        Matplotlib Figure object.
    """
    # Check if sigma column exists
    if sigma_col not in preds_df.columns:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"'{sigma_col}' column not found in predictions",
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Merge on object_id
    merged = preds_df.merge(
        truth_df[["object_id", true_col]],
        on="object_id",
        suffixes=("_pred", "_true")
    )

    if len(merged) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No matching data", ha='center', va='center',
                transform=ax.transAxes)
        return fig

    # Extract values
    if f"{true_col}_true" in merged.columns:
        true_vals = merged[f"{true_col}_true"].values
    else:
        true_vals = merged[true_col].values

    if pred_col in merged.columns:
        pred_vals = merged[pred_col].values
    elif f"{pred_col}_pred" in merged.columns:
        pred_vals = merged[f"{pred_col}_pred"].values
    else:
        pred_vals = merged[pred_col].values

    sigma_vals = merged[sigma_col].values

    # Compute alias-aware errors
    errors = []
    for p, t in zip(pred_vals, true_vals):
        if np.isfinite(p) and np.isfinite(t) and t > 0:
            errors.append(alias_aware_relative_error(p, t))
        else:
            errors.append(np.nan)
    errors = np.array(errors)

    # Filter to finite values
    mask = np.isfinite(errors) & np.isfinite(sigma_vals) & (sigma_vals > 0)
    errors_plot = errors[mask]
    sigma_plot = sigma_vals[mask]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if len(errors_plot) == 0:
        ax.text(0.5, 0.5, "No valid data points",
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Scatter plot
    ax.scatter(sigma_plot, errors_plot, alpha=0.5, s=20)

    # Log scale often works better for error distributions
    ax.set_xscale('log')
    ax.set_yscale('log')

    # Reference line (perfect calibration)
    x_ref = np.logspace(np.log10(sigma_plot.min()), np.log10(sigma_plot.max()), 50)
    ax.plot(x_ref, x_ref, 'r--', lw=1.5, label='Perfect calibration', alpha=0.7)

    # Labels
    ax.set_xlabel(f'Uncertainty ({sigma_col})', fontsize=12)
    ax.set_ylabel('Alias-aware relative error', fontsize=12)
    ax.set_title('Uncertainty vs Error', fontsize=14)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, which='both')

    # Compute correlation
    log_sigma = np.log10(sigma_plot)
    log_error = np.log10(errors_plot + 1e-10)
    corr = np.corrcoef(log_sigma, log_error)[0, 1]

    stats_text = f"N = {len(errors_plot)}\nlog-log corr = {corr:.3f}"
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')

    return fig


def plot_error_histogram(
    truth_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    out_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 5),
    true_col: str = "period_hours",
    pred_col: str = "period_hours",
    bins: int = 50,
    max_error: float = 0.5
) -> plt.Figure:
    """
    Plot histogram of prediction errors.

    Args:
        truth_df: DataFrame with object_id, period_hours (true).
        preds_df: DataFrame with object_id, period_hours (predicted).
        out_path: Optional path to save figure.
        figsize: Figure size tuple.
        true_col: Column name for true period.
        pred_col: Column name for predicted period.
        bins: Number of histogram bins.
        max_error: Maximum error to display.

    Returns:
        Matplotlib Figure object.
    """
    # Merge on object_id
    merged = preds_df.merge(
        truth_df[["object_id", true_col]],
        on="object_id",
        suffixes=("_pred", "_true")
    )

    if len(merged) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No matching data", ha='center', va='center',
                transform=ax.transAxes)
        return fig

    # Extract values
    if f"{true_col}_true" in merged.columns:
        true_vals = merged[f"{true_col}_true"].values
    else:
        true_vals = merged[true_col].values

    if pred_col in merged.columns:
        pred_vals = merged[pred_col].values
    elif f"{pred_col}_pred" in merged.columns:
        pred_vals = merged[f"{pred_col}_pred"].values
    else:
        pred_vals = merged[pred_col].values

    # Compute alias-aware errors
    errors = []
    for p, t in zip(pred_vals, true_vals):
        if np.isfinite(p) and np.isfinite(t) and t > 0:
            errors.append(alias_aware_relative_error(p, t))
        else:
            errors.append(np.nan)
    errors = np.array(errors)
    errors = errors[np.isfinite(errors)]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if len(errors) == 0:
        ax.text(0.5, 0.5, "No valid errors",
                ha='center', va='center', transform=ax.transAxes)
        return fig

    # Histogram
    ax.hist(errors, bins=bins, range=(0, max_error), edgecolor='black',
            alpha=0.7, color='steelblue')

    # Threshold lines
    for tol, color, label in [(0.05, 'green', '5%'), (0.10, 'orange', '10%')]:
        ax.axvline(tol, color=color, linestyle='--', lw=2, label=f'{label} threshold')

    ax.set_xlabel('Alias-aware relative error', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Period Errors', fontsize=14)

    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Stats
    median_err = np.median(errors)
    acc_5 = np.mean(errors <= 0.05)
    acc_10 = np.mean(errors <= 0.10)

    stats_text = (f"N = {len(errors)}\n"
                  f"Median = {median_err:.3f}\n"
                  f"Acc@5% = {acc_5*100:.1f}%\n"
                  f"Acc@10% = {acc_10*100:.1f}%")
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches='tight')

    return fig
