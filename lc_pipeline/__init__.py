"""
Lightcurve Period Prediction Pipeline.

A physics-based multi-epoch period estimation pipeline using
Lomb-Scargle periodograms and product-of-experts consensus.

Main Components:
- config: Configuration dataclasses (PeriodConfig, ColumnConfig)
- data: Data structures and I/O (LightcurveEpoch, AsteroidLightcurves)
- period_search: Lomb-Scargle implementation
- posterior: Posterior computation and credible intervals
- consensus: Multi-epoch consensus engine
- metrics: Evaluation metrics (alias-aware)
- io_utils: File I/O utilities
- plotting: Diagnostic visualizations

Example Usage:
    from lc_pipeline import (
        ConsensusEngine,
        PeriodConfig,
        load_manifest,
        group_epochs_by_object,
        evaluate_predictions,
    )

    # Load data
    manifest = load_manifest("data/manifest.csv")
    objects = group_epochs_by_object(manifest)

    # Run period estimation
    engine = ConsensusEngine()
    predictions = engine.predict_many(objects)

    # Evaluate
    truth = load_groundtruth("data/groundtruth.csv")
    metrics = evaluate_predictions(predictions, truth)
"""

__version__ = "1.0.0"

# Configuration
from .config import (
    PeriodConfig,
    ColumnConfig,
    PhysicalAliasConfig,
    DEFAULT_PERIOD_CONFIG,
    DEFAULT_COLUMN_CONFIG,
    DEFAULT_PHYSICAL_ALIAS_CONFIG,
)

# Data structures and I/O
from .data import (
    LightcurveEpoch,
    AsteroidLightcurves,
    flux_to_mag,
    load_manifest,
    load_groundtruth,
    load_epoch_from_file,
    load_epoch_from_row,
    group_epochs_by_object,
    extract_groundtruth_from_lightcurves,
)

# Period search
from .period_search import (
    lomb_scargle_period_search,
    inject_alias_candidates,
    sigma_clip,
)

# Posterior computation
from .posterior import (
    scores_to_probs,
    cluster_periods,
    aggregate_multi_epoch_posterior,
    compute_credible_interval,
    posterior_summary,
)

# Consensus engine
from .consensus import (
    ConsensusEngine,
    run_consensus_pipeline,
)

# Metrics
from .metrics import (
    relative_error,
    alias_aware_relative_error,
    accuracy_at_tol,
    evaluate_predictions,
    format_metrics_report,
)

# I/O utilities
from .io_utils import (
    ensure_dir,
    save_predictions_csv,
    load_predictions_csv,
    build_manifest_from_dir,
)

# Plotting
from .plotting import (
    plot_period_parity,
    plot_uncertainty_vs_error,
    plot_error_histogram,
)

__all__ = [
    # Version
    "__version__",
    # Config
    "PeriodConfig",
    "ColumnConfig",
    "PhysicalAliasConfig",
    "DEFAULT_PERIOD_CONFIG",
    "DEFAULT_COLUMN_CONFIG",
    "DEFAULT_PHYSICAL_ALIAS_CONFIG",
    # Data
    "LightcurveEpoch",
    "AsteroidLightcurves",
    "flux_to_mag",
    "load_manifest",
    "load_groundtruth",
    "load_epoch_from_file",
    "load_epoch_from_row",
    "group_epochs_by_object",
    "extract_groundtruth_from_lightcurves",
    # Period search
    "lomb_scargle_period_search",
    "inject_alias_candidates",
    "sigma_clip",
    # Posterior
    "scores_to_probs",
    "cluster_periods",
    "aggregate_multi_epoch_posterior",
    "compute_credible_interval",
    "posterior_summary",
    # Consensus
    "ConsensusEngine",
    "run_consensus_pipeline",
    # Metrics
    "relative_error",
    "alias_aware_relative_error",
    "accuracy_at_tol",
    "evaluate_predictions",
    "format_metrics_report",
    # I/O
    "ensure_dir",
    "save_predictions_csv",
    "load_predictions_csv",
    "build_manifest_from_dir",
    # Plotting
    "plot_period_parity",
    "plot_uncertainty_vs_error",
    "plot_error_histogram",
]
