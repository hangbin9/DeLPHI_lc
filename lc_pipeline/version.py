"""
lc_pipeline: Version Information

Provides version string for the asteroid lightcurve analysis pipeline.
"""

__version__ = "1.1.0"


def get_version() -> str:
    """Return current version string."""
    return __version__


def get_version_info() -> dict:
    """Return version metadata as dict."""
    return {
        'version': __version__,
        'description': 'DeLPHI asteroid lightcurve analysis pipeline',
        'status': 'production',
        'published_cv_sample_size': 174,
        'published_cv_mean_oracle_error_deg': 19.02,
        'published_cv_mean_oracle_error_std_deg': 2.68,
        'published_cv_pooled_median_oracle_error_deg': 16.61,
        'published_end_to_end_mean_oracle_error_deg': 18.90,
        'published_ztf_sample_size': 163,
        'published_ztf_mean_oracle_error_deg': 18.82,
        'published_ztf_mean_oracle_error_std_deg': 1.02,
        'published_period_error_median_relative': 0.053,
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
            'fold_0.pt',
            'fold_1.pt',
            'fold_2.pt',
            'fold_3.pt',
            'fold_4.pt',
        ],
    }
