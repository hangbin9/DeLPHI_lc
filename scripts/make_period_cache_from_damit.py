#!/usr/bin/env python3
"""
Build period cache from DAMIT shape-model spin solutions.

Discovers spin.txt files under a DAMIT root directory and extracts rotation
periods. Optionally augments from packed NPZ files that may contain period_hours.

Usage:
    python scripts/make_period_cache_from_damit.py \
        --damit-root /path/to/DAMIT \
        --out artifacts/period_cache_damit.json \
        --prefer-spin 1 \
        --npz-dir datasets/damit_real_multiepoch_v0 \
        --strict 1 \
        --min-frac 0.95
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pole_synth.period_cache import PeriodInfo, save_period_cache


def parse_spin_file(path: Path) -> Optional[float]:
    """
    Parse rotation period from a DAMIT spin.txt file.

    DAMIT spin.txt format (standard):
        Line 1: lambda beta period (ecliptic coords in degrees, period in hours)
        Line 2: epoch and uncertainty info
        Line 3: shape model parameters

    We prioritize the first line, third value as the period.
    Falls back to heuristics if the format doesn't match.

    Args:
        path: Path to spin.txt file

    Returns:
        Period in hours, or None if parsing failed
    """
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    lines = content.strip().split('\n')
    if not lines:
        return None

    # Primary approach: Parse first line as "lambda beta period"
    float_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    first_line_matches = re.findall(float_pattern, lines[0])

    first_line_floats = []
    for m in first_line_matches:
        try:
            val = float(m)
            first_line_floats.append(val)
        except ValueError:
            continue

    # If first line has exactly 3 values and third is plausible period, use it
    if len(first_line_floats) >= 3:
        period_candidate = first_line_floats[2]  # Third value is period
        if 0.1 <= period_candidate <= 10000:  # Reasonable period range
            return period_candidate

    # Fallback: extract all floats and use heuristics
    matches = re.findall(float_pattern, content)
    floats = []
    for m in matches:
        try:
            val = float(m)
            if val != 0 and not (val != val):  # not zero, not NaN
                floats.append(val)
        except ValueError:
            continue

    if not floats:
        return None

    # Filter to plausible period range (hours)
    # Most asteroid periods are 2-20 hours, but can range 0.5-2000
    plausible = [f for f in floats if 0.5 <= f <= 2000]

    if plausible:
        # Prefer the first plausible value (from first line's third position typically)
        period_hours = plausible[0]
    else:
        # Check if values might be in days (< 2.0)
        # Typical asteroid day is 2-20 hours = 0.08-0.83 days
        days_candidates = [f for f in floats if 0.05 <= f <= 100]
        if days_candidates:
            # Assume days if small values, convert to hours
            period_days = days_candidates[0]
            if period_days < 2.0:
                # Likely in days, convert to hours
                period_hours = period_days * 24.0
            else:
                period_hours = period_days  # Assume already hours
        else:
            # Last resort: use first float
            period_hours = floats[0]

    # Sanity check final value
    if period_hours <= 0 or period_hours > 10000:
        return None

    return period_hours


def extract_object_id_from_path(spin_path: Path) -> Optional[str]:
    """
    Extract object_id from spin.txt path.

    Expected patterns:
        .../files/asteroid_<asteroid_id>/model_<model_id>/spin.txt  (actual DAMIT structure)
        .../DAMIT/shape_models/<asteroid_id>/<model_id>/spin.txt
        .../damit-<date>/<asteroid_id>/<model_id>/spin.txt
        .../<asteroid_id>/<model_id>/spin.txt

    Returns:
        object_id like "asteroid_101_model_101" (matching CSV naming) or None
    """
    parts = spin_path.parts

    # Need at least 2 parent directories
    if len(parts) < 3:
        return None

    # Parent is model dir, grandparent is asteroid dir
    model_dir = parts[-2]
    asteroid_dir = parts[-3]

    # Pattern 1: asteroid_<id>/model_<id> (actual DAMIT structure)
    asteroid_match = re.match(r"asteroid_(\d+)", asteroid_dir)
    model_match = re.match(r"model_(\d+)", model_dir)

    if asteroid_match and model_match:
        asteroid_id = asteroid_match.group(1)
        model_id = model_match.group(1)
        # Return in CSV naming format: asteroid_<aid>_model_<mid>
        return f"asteroid_{asteroid_id}_model_{model_id}"

    # Pattern 2: simple numeric IDs
    if asteroid_dir.isdigit() and model_dir.isdigit():
        return f"asteroid_{asteroid_dir}_model_{model_dir}"

    # Pattern 3: alphanumeric fallback
    if (len(model_dir) < 20 and model_dir.replace("_", "").isalnum() and
        len(asteroid_dir) < 20 and asteroid_dir.replace("_", "").isalnum()):
        return f"asteroid_{asteroid_dir}_model_{model_dir}"

    return None


def discover_spin_files(damit_root: Path) -> Dict[str, Path]:
    """
    Discover all spin.txt files under DAMIT root.

    Returns:
        Dict mapping object_id to spin.txt path
    """
    results = {}
    warnings = []

    for spin_path in damit_root.rglob("spin.txt"):
        object_id = extract_object_id_from_path(spin_path)
        if object_id is None:
            warnings.append(f"Could not extract object_id from: {spin_path}")
            continue

        if object_id in results:
            # Prefer the more recent/deeper path
            pass
        results[object_id] = spin_path

    if warnings:
        print(f"Warnings during discovery ({len(warnings)}):")
        for w in warnings[:10]:
            print(f"  {w}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")

    return results


def load_periods_from_npz_dir(npz_dir: Path) -> Dict[str, float]:
    """
    Load periods from packed NPZ files that contain period_hours.

    NPZ filename pattern: damit_{asteroid_id}_{model_id}.npz

    Returns:
        Dict mapping object_id to period_hours
    """
    results = {}

    for npz_path in npz_dir.glob("*.npz"):
        # Extract object_id from filename
        stem = npz_path.stem
        if stem.startswith("damit_"):
            object_id = stem
        else:
            # Try to parse pattern
            match = re.match(r"(damit_\d+_\d+)", stem)
            if match:
                object_id = match.group(1)
            else:
                continue

        try:
            data = np.load(npz_path, allow_pickle=True)

            # Check for period fields
            period_hours = None
            for key in ["period_hours", "P_hours", "period", "P"]:
                if key in data:
                    val = data[key]
                    if isinstance(val, np.ndarray):
                        val = float(val.item()) if val.size == 1 else float(val[0])
                    else:
                        val = float(val)

                    # Convert if needed (assume P without suffix is hours)
                    if key in ["period", "P"] and val < 2.0:
                        # Might be days
                        val = val * 24.0

                    period_hours = val
                    break

            if period_hours is not None and 0.5 <= period_hours <= 2000:
                results[object_id] = period_hours

        except Exception as e:
            print(f"Warning: Could not load {npz_path}: {e}")
            continue

    return results


def get_object_ids_from_npz_dir(npz_dir: Path) -> List[str]:
    """Get list of object_ids from NPZ filenames."""
    object_ids = []
    for npz_path in npz_dir.glob("*.npz"):
        stem = npz_path.stem
        if stem.startswith("damit_"):
            object_ids.append(stem)
        elif stem.startswith("asteroid_"):
            # Format: asteroid_<aid>_model_<mid>
            object_ids.append(stem)
        else:
            match = re.match(r"(damit_\d+_\d+)", stem)
            if match:
                object_ids.append(match.group(1))
    return object_ids


def get_object_ids_from_csv_dir(csv_dir: Path) -> List[str]:
    """Get list of object_ids from CSV filenames."""
    object_ids = []
    for csv_path in csv_dir.glob("*.csv"):
        stem = csv_path.stem
        # Format: asteroid_<aid>_model_<mid>
        if stem.startswith("asteroid_") and "_model_" in stem:
            object_ids.append(stem)
    return object_ids


def main():
    parser = argparse.ArgumentParser(
        description="Build period cache from DAMIT spin solutions"
    )
    parser.add_argument(
        "--damit-root",
        type=Path,
        help="Root directory containing DAMIT shape models with spin.txt files",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path for period cache JSON",
    )
    parser.add_argument(
        "--prefer-spin",
        type=int,
        default=1,
        help="Prefer spin.txt over NPZ periods (1=yes, 0=no)",
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        help="Optional: Directory with packed NPZ files to augment/validate",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        help="Optional: Directory with CSV files to validate coverage against",
    )
    parser.add_argument(
        "--strict",
        type=int,
        default=1,
        help="Require minimum coverage (1=yes, 0=no)",
    )
    parser.add_argument(
        "--min-frac",
        type=float,
        default=0.95,
        help="Minimum coverage fraction when --strict=1",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Building Period Cache from DAMIT")
    print("=" * 60)

    periods: Dict[str, float] = {}

    # Step 1: Discover and parse spin.txt files
    if args.damit_root and args.damit_root.exists():
        print(f"\nDiscovering spin.txt files under: {args.damit_root}")
        spin_files = discover_spin_files(args.damit_root)
        print(f"Found {len(spin_files)} spin.txt files")

        parsed = 0
        for object_id, spin_path in spin_files.items():
            period_hours = parse_spin_file(spin_path)
            if period_hours is not None:
                periods[object_id] = period_hours
                parsed += 1

        print(f"Successfully parsed {parsed}/{len(spin_files)} periods from spin.txt")
    else:
        if args.damit_root:
            print(f"Warning: DAMIT root not found: {args.damit_root}")

    # Step 2: Augment from NPZ files
    if args.npz_dir and args.npz_dir.exists():
        print(f"\nLoading periods from NPZ files in: {args.npz_dir}")
        npz_periods = load_periods_from_npz_dir(args.npz_dir)
        print(f"Found {len(npz_periods)} periods in NPZ files")

        if args.prefer_spin:
            # Only add NPZ periods for objects not in spin.txt
            added = 0
            for object_id, period_hours in npz_periods.items():
                if object_id not in periods:
                    periods[object_id] = period_hours
                    added += 1
            print(f"Added {added} periods from NPZ (not in spin.txt)")
        else:
            # NPZ overrides spin.txt
            overridden = sum(1 for oid in npz_periods if oid in periods)
            periods.update(npz_periods)
            print(f"Used NPZ periods ({overridden} overrode spin.txt)")

    # Step 3: Build cache
    print(f"\nTotal periods collected: {len(periods)}")

    cache = {oid: PeriodInfo(period_hours=ph) for oid, ph in periods.items()}

    # Step 4: Validate coverage if NPZ or CSV dir provided
    target_ids = []
    if args.npz_dir and args.npz_dir.exists():
        target_ids = get_object_ids_from_npz_dir(args.npz_dir)
        print(f"\nFound {len(target_ids)} objects in NPZ dir")
    elif args.csv_dir and args.csv_dir.exists():
        target_ids = get_object_ids_from_csv_dir(args.csv_dir)
        print(f"\nFound {len(target_ids)} objects in CSV dir")

    if target_ids:
        loaded = sum(1 for oid in target_ids if oid in cache)
        total = len(target_ids)
        frac = loaded / total if total > 0 else 1.0

        print(f"Periods loaded: {loaded}/{total} ({frac:.1%})")

        if args.strict and frac < args.min_frac:
            print(f"ERROR: Coverage {frac:.1%} < required {args.min_frac:.1%}")
            print("Missing objects:")
            missing = [oid for oid in target_ids if oid not in cache]
            for oid in missing[:20]:
                print(f"  {oid}")
            if len(missing) > 20:
                print(f"  ... and {len(missing) - 20} more")
            sys.exit(1)

    # Step 5: Save cache
    save_period_cache(args.out, cache)
    print(f"\nSaved period cache to: {args.out}")
    print(f"Total entries: {len(cache)}")

    # Print sample entries
    print("\nSample entries:")
    for i, (oid, info) in enumerate(list(cache.items())[:5]):
        print(f"  {oid}: {info.period_hours:.4f} hours")

    print("\nDone.")


if __name__ == "__main__":
    main()
