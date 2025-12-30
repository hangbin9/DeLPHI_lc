#!/usr/bin/env python3
"""
Build a manifest CSV from a directory of lightcurve files.

Usage:
    python -m scripts.build_manifest_from_dir \
        --root-dir DAMIT_csv_high \
        --out-manifest data/damit_manifest.csv

This script walks through the specified directory, finds CSV files,
and infers object_id and epoch_id from filenames.

For DAMIT-style filenames like "asteroid_101_model_101.csv":
    - object_id = "asteroid_101"
    - epoch_id = "model_101"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lc_pipeline.io_utils import build_manifest_from_dir, ensure_dir


def main():
    parser = argparse.ArgumentParser(
        description="Build manifest CSV from directory of lightcurve files."
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default="DAMIT_csv_high",
        help="Root directory containing lightcurve CSV files."
    )
    parser.add_argument(
        "--out-manifest",
        type=str,
        default="data/damit_manifest.csv",
        help="Output path for manifest CSV."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="Glob pattern for matching files (default: *.csv)."
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't search subdirectories."
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    out_path = Path(args.out_manifest)

    if not root_dir.exists():
        print(f"Error: Directory not found: {root_dir}")
        sys.exit(1)

    print(f"Scanning directory: {root_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Recursive: {not args.no_recursive}")

    # Build manifest
    manifest_df = build_manifest_from_dir(
        root_dir=root_dir,
        pattern=args.pattern,
        recursive=not args.no_recursive
    )

    print(f"\nFound {len(manifest_df)} files")
    print(f"Unique objects: {manifest_df['object_id'].nunique()}")

    # Save
    ensure_dir(out_path.parent)
    manifest_df.to_csv(out_path, index=False)
    print(f"\nManifest saved to: {out_path}")

    # Show sample
    print("\nSample entries:")
    print(manifest_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
