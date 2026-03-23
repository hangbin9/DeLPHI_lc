"""
I/O utilities for the period pipeline.

This module provides:
- save_predictions_csv: save predictions DataFrame
- load_predictions_csv: load predictions DataFrame
- ensure_dir: create directory if needed
- build_manifest_from_dir: auto-generate manifest from directory structure
"""

import os
import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        Path object for the directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_predictions_csv(
    df: pd.DataFrame,
    path: Union[str, Path]
) -> None:
    """
    Save predictions DataFrame to CSV.

    Args:
        df: DataFrame with prediction results.
        path: Output file path.
    """
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def load_predictions_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load predictions DataFrame from CSV.

    Args:
        path: Path to predictions CSV.

    Returns:
        DataFrame with predictions.

    Raises:
        FileNotFoundError: If file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")
    return pd.read_csv(path)


def parse_damit_filename(filename: str) -> Tuple[str, str]:
    """
    Parse DAMIT-style filename to extract object_id and epoch_id.

    Expected patterns:
    - asteroid_101_model_101.csv -> object_id="asteroid_101", epoch_id="model_101"
    - asteroid_1017_model_1743.csv -> object_id="asteroid_1017", epoch_id="model_1743"

    Args:
        filename: Filename (without directory path).

    Returns:
        (object_id, epoch_id) tuple.
    """
    # Remove .csv extension
    name = filename.replace(".csv", "")

    # Try pattern: asteroid_XXX_model_YYY
    match = re.match(r"(asteroid_\d+)_(.+)", name)
    if match:
        return match.group(1), match.group(2)

    # Fallback: split on first underscore
    parts = name.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]

    # Last resort: use full name as object_id
    return name, "epoch_0"


def parse_generic_filename(filename: str) -> Tuple[str, str]:
    """
    Generic filename parser: uses filename stem as object_id.

    Args:
        filename: Filename (without directory path).

    Returns:
        (object_id, epoch_id) tuple.
    """
    name = Path(filename).stem
    return name, "epoch_0"


def build_manifest_from_dir(
    root_dir: Union[str, Path],
    pattern: str = "*.csv",
    parser: Optional[Callable[[str], Tuple[str, str]]] = None,
    recursive: bool = True
) -> pd.DataFrame:
    """
    Build a manifest DataFrame from a directory of lightcurve files.

    Walks through the directory, finds matching files, and infers
    object_id and epoch_id from filenames.

    Args:
        root_dir: Root directory to scan.
        pattern: Glob pattern for matching files (default: "*.csv").
        parser: Function to parse filename -> (object_id, epoch_id).
               If None, uses parse_damit_filename.
        recursive: Whether to search subdirectories.

    Returns:
        DataFrame with columns: object_id, epoch_id, file_path
    """
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    if parser is None:
        parser = parse_damit_filename

    # Find all matching files
    if recursive:
        files = list(root_dir.rglob(pattern))
    else:
        files = list(root_dir.glob(pattern))

    records: List[dict] = []
    for file_path in sorted(files):
        filename = file_path.name
        object_id, epoch_id = parser(filename)

        records.append({
            "object_id": object_id,
            "epoch_id": epoch_id,
            "file_path": str(file_path.absolute()),
        })

    return pd.DataFrame(records)


def build_manifest_from_nested_dir(
    root_dir: Union[str, Path],
    pattern: str = "*.csv"
) -> pd.DataFrame:
    """
    Build manifest from nested directory structure.

    Assumes structure: root_dir/object_id/epoch_id.csv

    Args:
        root_dir: Root directory containing object subdirectories.
        pattern: Glob pattern for matching files.

    Returns:
        DataFrame with columns: object_id, epoch_id, file_path
    """
    root_dir = Path(root_dir)

    records: List[dict] = []
    for object_dir in sorted(root_dir.iterdir()):
        if not object_dir.is_dir():
            continue

        object_id = object_dir.name

        for file_path in sorted(object_dir.glob(pattern)):
            epoch_id = file_path.stem

            records.append({
                "object_id": object_id,
                "epoch_id": epoch_id,
                "file_path": str(file_path.absolute()),
            })

    return pd.DataFrame(records)
