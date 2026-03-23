"""
lc_pipeline: Version Information

Provides version string for the asteroid lightcurve analysis pipeline.
"""

__version__ = "0.1.0"


def get_version() -> str:
    """Return current version string."""
    return __version__


def get_version_info() -> dict:
    """Return version metadata as dict."""
    return {
        'version': __version__,
        'description': 'DeLPHI: Deep Learning Photometry-based Hypothesis Inference',
        'status': 'pre-release',
        'validation_date': '2026-03-23',
        'oracle_error_mean': 19.02,  # degrees (5-fold CV, 174 asteroids, asteroid-level)
        'oracle_error_std': 2.68,  # degrees (across-fold std)
        'oracle_error_median_pooled': 16.61,  # degrees (pooled across all folds)
        'ztf_external_mean': 18.82,  # degrees (163 external asteroids)
        'period_error_median': 0.053,  # relative (alias-aware, 174/174 success)
        'period_accuracy_10pct': 0.55,  # fraction within 10%
        'n_params': 994185,
        'features': [
            'multi_epoch_period_estimation',
            'pole_prediction_period_aware',
            '13_feature_tokenization',
            'uncertainty_quantification',
            'k3_multi_hypothesis',
            'single_epoch_training',
            'scientifically_rigorous_oracle',
            '5fold_cross_validation',
        ],
        'checkpoints': [
            'fold_0.pt (oracle=19.51° mean, 36 val asteroids)',
            'fold_1.pt (oracle=14.88° mean, 36 val asteroids)',
            'fold_2.pt (oracle=18.32° mean, 35 val asteroids)',
            'fold_3.pt (oracle=22.05° mean, 35 val asteroids, seed=42)',
            'fold_4.pt (oracle=20.34° mean, 35 val asteroids)',
        ],
    }
