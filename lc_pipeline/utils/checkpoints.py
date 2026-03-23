"""Checkpoint inspection and validation."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)


def inspect_checkpoint(path: Path) -> Dict:
    """
    Inspect checkpoint without loading into model.

    Args:
        path: Path to checkpoint file

    Returns:
        Dict with keys: k, has_quality_head, model_type, axisnet_version, config_hash
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location='cpu')
    config = checkpoint.get('config', {})

    inspection = {
        'k': config.get('k', None),
        'has_quality_head': config.get('has_quality_head', False),
        'model_type': config.get('model_type', 'unknown'),
        'axisnet_version': config.get('axisnet_version', 'unknown'),
        'config_hash': hash(str(config)),
    }

    return inspection


def validate_checkpoint_for_inference(
    path: Path,
    selector_mode: str = "quality_head",
) -> Tuple[bool, str]:
    """
    Validate checkpoint compatibility for inference.

    Args:
        path: Path to checkpoint
        selector_mode: Intended selector mode

    Returns:
        (is_valid, message)
    """
    try:
        inspection = inspect_checkpoint(path)
    except Exception as e:
        return False, f"Failed to inspect checkpoint: {e}"

    # Check required fields
    if inspection['k'] is None:
        return False, "Checkpoint missing 'k' (number of hypotheses)"

    if inspection['model_type'] != 'AxisNetK2QualityModel':
        return False, f"Unknown model type: {inspection['model_type']}"

    # Check selector compatibility
    if selector_mode == "quality_head":
        if not inspection['has_quality_head']:
            return False, "Selector 'quality_head' requires model with quality head"

    return True, "Checkpoint valid"


def get_default_selector_for_checkpoint(path: Path) -> str:
    """
    Determine default selector based on checkpoint features.

    Args:
        path: Path to checkpoint

    Returns:
        Selector mode: 'quality_head' if available, else 'naive0'
    """
    try:
        inspection = inspect_checkpoint(path)
        if inspection['has_quality_head']:
            return "quality_head"
        else:
            return "naive0"
    except Exception as e:
        logger.warning(f"Failed to determine selector, defaulting to naive0: {e}")
        return "naive0"
