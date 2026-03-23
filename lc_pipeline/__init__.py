"""
End-to-End Asteroid Lightcurve Analysis Pipeline.

A complete pipeline for asteroid lightcurve analysis combining:
1. Period prediction: Physics-based multi-epoch estimation using Lomb-Scargle
2. Pole prediction: Deep learning-based pole orientation estimation with period alias forking

Main Components:

Period Prediction (Part 1):
- config: Configuration dataclasses (PeriodConfig, ColumnConfig)
- data: Data structures and I/O (LightcurveEpoch, AsteroidLightcurves)
- period_search: Lomb-Scargle implementation
- posterior: Posterior computation and credible intervals
- consensus: Multi-epoch consensus engine
- metrics: Evaluation metrics (alias-aware)

Pole Prediction (Part 2):
- schema: Output dataclasses (AnalysisResult, PoleCandidate, PeriodResult)
- model: Hierarchical transformer architecture (PolePredictor)
- tokenizer: Lightcurve tokenization for neural networks
- forking: Period alias expansion strategy
- pole: Pole inference engine
- pipeline: End-to-end orchestration

Example Usage (End-to-End):
    from lc_pipeline import analyze

    # Analyze asteroid with DAMIT format epochs
    result = analyze(epochs, "asteroid_1017", period_hours=8.5)

    print(f"Period: {result.period.period_hours:.2f} h")
    print(f"Best pole: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")

    # All 9 candidates (3 periods × 3 slots)
    for pole in result.poles:
        print(f"  λ={pole.lambda_deg:.1f}°, β={pole.beta_deg:.1f}°, "
              f"P={pole.alias} ({pole.period_hours:.2f}h), score={pole.score:.3f}")

Example Usage (Period Only):
    from lc_pipeline import ConsensusEngine, load_manifest, group_epochs_by_object

    manifest = load_manifest("data/manifest.csv")
    objects = group_epochs_by_object(manifest)

    engine = ConsensusEngine()
    predictions = engine.predict_many(objects)
"""

# Version
from .version import __version__

# Configuration
from .period.config import (
    PeriodConfig,
    ColumnConfig,
    PhysicalAliasConfig,
    DEFAULT_PERIOD_CONFIG,
    DEFAULT_COLUMN_CONFIG,
    DEFAULT_PHYSICAL_ALIAS_CONFIG,
)

# Data structures and I/O
from .data.loaders import (
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
from .period.period_search import (
    lomb_scargle_period_search,
    inject_alias_candidates,
    sigma_clip,
)

# Posterior computation
from .period.posterior import (
    scores_to_probs,
    cluster_periods,
    aggregate_multi_epoch_posterior,
    compute_credible_interval,
    posterior_summary,
)

# Consensus engine
from .period.consensus import (
    ConsensusEngine,
    run_consensus_pipeline,
)

# Metrics
from .evaluation.metrics import (
    relative_error,
    alias_aware_relative_error,
    accuracy_at_tol,
    evaluate_predictions,
    format_metrics_report,
)

# I/O utilities
from .utils.io import (
    ensure_dir,
    save_predictions_csv,
    load_predictions_csv,
    build_manifest_from_dir,
)

# Plotting
from .utils.plotting import (
    plot_period_parity,
    plot_uncertainty_vs_error,
    plot_error_histogram,
)

# Pole prediction (Part 2)
from .inference.schema import (
    PoleCandidate,
    PeriodResult,
    PoleUncertainty,
    AnalysisResult,
)

from .inference.coordinates import (
    xyz_to_ecliptic,
    ecliptic_to_xyz,
)

from .inference.pole import (
    PoleInference,
    PoleConfig,
)

from .inference.forking import (
    PeriodForker,
)

from .inference.uncertainty import (
    compute_uncertainty,
)

from .inference.pipeline import (
    LightcurvePipeline,
    analyze,
)

# Unified data format (schema and converters)
from .schema import (
    LightcurveData,
    Epoch,
    Observation,
    GroundTruth,
    PoleSolution,
    SimplifiedCSVSchema,
)

from .converters import (
    convert_damit_to_unified,
    load_damit_object,
    load_unified_json,
    load_unified_csv,
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
    # Pole prediction
    "PoleCandidate",
    "PeriodResult",
    "PoleUncertainty",
    "AnalysisResult",
    "xyz_to_ecliptic",
    "ecliptic_to_xyz",
    "PoleInference",
    "PoleConfig",
    "PeriodForker",
    "compute_uncertainty",
    "LightcurvePipeline",
    "analyze",
    # Unified data format
    "LightcurveData",
    "Epoch",
    "Observation",
    "GroundTruth",
    "PoleSolution",
    "SimplifiedCSVSchema",
    "convert_damit_to_unified",
    "load_damit_object",
    "load_unified_json",
    "load_unified_csv",
]
