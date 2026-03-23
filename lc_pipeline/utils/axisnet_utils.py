"""Utility functions for AxisNet."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .io import ensure_dir


def stable_hash_to_fold(object_id: str, num_folds: int = 3) -> int:
    """
    Deterministically map object_id to fold using stable hash.

    Args:
        object_id: String identifier for object (e.g., asteroid name)
        num_folds: Number of folds (default 3)

    Returns:
        Fold index (0 to num_folds-1)
    """
    h = hashlib.sha1(object_id.encode('utf-8')).hexdigest()
    fold = int(h, 16) % num_folds
    return fold


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)


def save_json(data: Any, path: Path) -> None:
    """Save data to JSON file, handling numpy types."""
    ensure_dir(path.parent)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def load_json(path: Path) -> Any:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)


def dict_hash(d: Dict) -> str:
    """Compute stable hash of dictionary."""
    json_str = json.dumps(d, sort_keys=True)
    return hashlib.sha1(json_str.encode('utf-8')).hexdigest()[:8]
