"""
Phase 29I: Version Information.

Provides version string for release tracking and reproducibility.
"""

__version__ = "29I-rc1"

# Version history:
# 29I-rc1: First release candidate with:
#   - K=2 multi-hypothesis pole regressor
#   - Learned hypothesis selector (Phase 29H)
#   - CV evaluation framework (Phase 29G)
#   - One-command paper run (Phase 29I)


def get_version() -> str:
    """Return current version string."""
    return __version__


def get_version_info() -> dict:
    """Return version metadata as dict."""
    return {
        'version': __version__,
        'phase': '29I',
        'release_type': 'rc1',
        'features': [
            'multi_hypothesis_k2',
            'learned_selector',
            'cv_evaluation',
            'paper_run',
        ],
    }
