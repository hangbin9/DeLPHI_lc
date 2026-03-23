#!/usr/bin/env python3
"""
run_pole_prediction.py - Run pole prediction on asteroid lightcurve data.

This script provides an easy-to-use interface for running pole predictions on
asteroid lightcurve data using trained DeLPHI checkpoints.

Usage Examples:
    # Simple usage with provided checkpoints
    python run_pole_prediction.py --input asteroid_101.csv

    # With custom checkpoint
    python run_pole_prediction.py --input asteroid_101.csv --checkpoint my_model.pt

    # Batch processing
    python run_pole_prediction.py --input-dir my_asteroids/ --output results.json

    # With known period (skip period estimation)
    python run_pole_prediction.py --input asteroid_101.csv --period 8.5

Input Data Formats:
    1. DAMIT CSV format (8 columns):
       time_jd, brightness, sun_x, sun_y, sun_z, obs_x, obs_y, obs_z

    2. DAMIT lc.json format:
       [{"object_id": "...", "epoch_jd": ..., "points": "..."}]

Output Formats:
    - console: Pretty-printed results (default)
    - json: JSON file with all results
    - csv: CSV file with top predictions
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd

try:
    from lc_pipeline import analyze, LightcurvePipeline
    from lc_pipeline.inference.schema import AnalysisResult
    from lc_pipeline.inference.pole import PoleConfig
except ImportError as e:
    print(f"ERROR: Failed to import lc_pipeline: {e}")
    print("\nPlease install lc_pipeline first:")
    print("  pip install -e .")
    sys.exit(1)


def load_damit_csv(csv_path: Path) -> List[np.ndarray]:
    """
    Load DAMIT CSV format lightcurve data.

    CSV format: time_jd, brightness, sun_x, sun_y, sun_z, obs_x, obs_y, obs_z

    Returns:
        List of epoch arrays with shape (N, 8)
    """
    try:
        # Use pandas to handle missing values and comments
        df = pd.read_csv(csv_path, comment='#')

        # Drop rows with any missing values
        df = df.dropna()

        # Convert to numpy array
        data = df.values

        if data.shape[1] != 8:
            raise ValueError(f"Expected 8 columns, got {data.shape[1]}")

        # For CSV files, treat all data as single epoch
        return [data]

    except Exception as e:
        raise ValueError(f"Failed to load CSV {csv_path}: {e}")


def load_damit_json(json_path: Path) -> List[np.ndarray]:
    """
    Load JSON format (DAMIT legacy or unified format).

    Supports two formats:

    1. DAMIT legacy format:
    [
      {
        "object_id": "...",
        "epoch_jd": ...,
        "points": "JD brightness sun_x sun_y sun_z obs_x obs_y obs_z\\n..."
      }
    ]

    2. Unified format (see docs/DATA_FORMAT.md):
    {
      "format_version": "1.0",
      "object_id": "...",
      "epochs": [
        {
          "epoch_id": 0,
          "observations": [
            {"time_jd": ..., "relative_brightness": ..., ...}
          ]
        }
      ]
    }

    Returns:
        List of epoch arrays with shape (N, 8)
    """
    try:
        with open(json_path) as f:
            data = json.load(f)

        epochs = []

        # Check for unified format (has "epochs" key with observations)
        if isinstance(data, dict) and 'epochs' in data:
            # Unified format
            for epoch in data['epochs']:
                obs_list = epoch.get('observations', [])
                if not obs_list:
                    continue

                points = []
                for obs in obs_list:
                    sun_vec = obs.get('sun_asteroid_vector', [0, 0, 0])
                    earth_vec = obs.get('earth_asteroid_vector', [0, 0, 0])
                    points.append([
                        obs['time_jd'],
                        obs['relative_brightness'],
                        sun_vec[0], sun_vec[1], sun_vec[2],
                        earth_vec[0], earth_vec[1], earth_vec[2]
                    ])

                if points:
                    epochs.append(np.array(points))
        else:
            # Legacy DAMIT format (list of epochs)
            for epoch in data:
                points_str = epoch['points'].strip()
                points = np.array([
                    list(map(float, line.split()))
                    for line in points_str.split('\n')
                    if line.strip()
                ])
                epochs.append(points)

        return epochs

    except Exception as e:
        raise ValueError(f"Failed to load JSON {json_path}: {e}")


def load_lightcurve(input_path: Path) -> List[np.ndarray]:
    """
    Load lightcurve data from CSV or JSON format.

    Args:
        input_path: Path to .csv or .json file

    Returns:
        List of epoch arrays
    """
    if input_path.suffix.lower() == '.csv':
        return load_damit_csv(input_path)
    elif input_path.suffix.lower() == '.json':
        return load_damit_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .csv or .json")


def format_result_console(result: AnalysisResult) -> str:
    """
    Format analysis result for console output.

    Args:
        result: AnalysisResult from analyze()

    Returns:
        Formatted string for display
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"POLE PREDICTION RESULTS: {result.object_id}")
    lines.append("=" * 80)
    lines.append("")

    # Period
    period = result.period
    lines.append(f"Period: {period.period_hours:.2f} ± {period.uncertainty_hours:.2f} h")
    lines.append("")

    # Best pole (with antipode - poles are ambiguous up to 180°)
    best = result.best_pole
    lines.append("Best Pole (or antipode):")
    lines.append(f"  λ = {best.lambda_deg:.1f}°, β = {best.beta_deg:.1f}°")
    lines.append(f"  Antipode: λ = {best.antipode_lambda_deg:.1f}°, β = {best.antipode_beta_deg:.1f}°")
    lines.append(f"  Quality score: {best.score:.3f}")
    lines.append(f"  Period alias: {best.alias}")
    lines.append("")

    # All candidates
    lines.append(f"All {len(result.poles)} candidates (3 periods × 3 poles):")
    lines.append(f"{'#':<3} {'Alias':<8} {'Period':<9} {'λ':<8} {'β':<8} {'Score':<6}")
    lines.append("-" * 50)

    for i, pole in enumerate(result.poles, 1):
        lines.append(
            f"{i:<3} {pole.alias:<8} {pole.period_hours:>7.2f}h "
            f"{pole.lambda_deg:>6.1f}° {pole.beta_deg:>6.1f}° {pole.score:>6.3f}"
        )

    lines.append("")

    # Uncertainty
    unc = result.uncertainty
    lines.append("Uncertainty:")
    lines.append(f"  Spread: {unc.spread_deg:.1f}°")
    lines.append(f"  Confidence: {unc.confidence:.2f}")
    lines.append("=" * 80)

    return "\n".join(lines)


def format_result_json(result: AnalysisResult) -> Dict[str, Any]:
    """
    Format analysis result as JSON dictionary.

    Args:
        result: AnalysisResult from analyze()

    Returns:
        Dictionary suitable for JSON serialization
    """
    def to_python(val):
        """Convert numpy types to Python types for JSON serialization."""
        if hasattr(val, 'item'):
            return val.item()
        return val

    return {
        "object_id": result.object_id,
        "period": {
            "period_hours": to_python(result.period.period_hours),
            "uncertainty_hours": to_python(result.period.uncertainty_hours),
            "ci_low_hours": to_python(result.period.ci_low_hours),
            "ci_high_hours": to_python(result.period.ci_high_hours),
            "n_epochs": result.period.n_epochs,
            "success": result.period.success,
        },
        "best_pole": {
            "lambda_deg": to_python(result.best_pole.lambda_deg),
            "beta_deg": to_python(result.best_pole.beta_deg),
            "xyz": [to_python(x) for x in result.best_pole.xyz],
            "period_hours": to_python(result.best_pole.period_hours),
            "alias": result.best_pole.alias,
            "score": to_python(result.best_pole.score),
            "slot": result.best_pole.slot,
        },
        "all_poles": [
            {
                "lambda_deg": to_python(pole.lambda_deg),
                "beta_deg": to_python(pole.beta_deg),
                "xyz": [to_python(x) for x in pole.xyz],
                "period_hours": to_python(pole.period_hours),
                "alias": pole.alias,
                "score": to_python(pole.score),
                "slot": pole.slot,
            }
            for pole in result.poles
        ],
        "uncertainty": {
            "spread_deg": to_python(result.uncertainty.spread_deg),
            "confidence": to_python(result.uncertainty.confidence),
        }
    }


def run_single(
    input_path: Path,
    object_id: Optional[str] = None,
    period_hours: Optional[float] = None,
    fold: int = 0,
    checkpoint_path: Optional[Path] = None,
) -> AnalysisResult:
    """
    Run pole prediction on a single asteroid.

    Args:
        input_path: Path to input file (.csv or .json)
        object_id: Object identifier (defaults to filename)
        period_hours: Known period in hours (optional, skips estimation)
        fold: Fold number if using built-in checkpoints (0-4)
        checkpoint_path: Path to custom checkpoint (optional)

    Returns:
        AnalysisResult with predictions
    """
    if object_id is None:
        object_id = input_path.stem

    print(f"Loading lightcurve: {input_path}")
    epochs = load_lightcurve(input_path)
    print(f"  Loaded {len(epochs)} epoch(s) with {sum(len(e) for e in epochs)} total observations")

    print(f"Running analysis...")
    if checkpoint_path:
        import tempfile
        import shutil
        print(f"  Using custom checkpoint: {checkpoint_path}")
        # Stage checkpoint as fold_0.pt in a temp directory so the pipeline can load it
        tmpdir = Path(tempfile.mkdtemp())
        shutil.copy2(checkpoint_path, tmpdir / "fold_0.pt")
        config = PoleConfig(checkpoint_dir=tmpdir)
        pipeline = LightcurvePipeline(pole_config=config)
        result = pipeline.analyze(
            epochs=epochs,
            object_id=object_id,
            period_hours=period_hours,
            fold=0,
        )
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        result = analyze(
            epochs=epochs,
            object_id=object_id,
            period_hours=period_hours,
            fold=fold,
        )

    return result


def run_batch(
    input_dir: Path,
    period_hours: Optional[float] = None,
    fold: int = 0,
    checkpoint_path: Optional[Path] = None,
) -> List[AnalysisResult]:
    """
    Run pole prediction on multiple asteroids in a directory.

    Args:
        input_dir: Directory containing .csv or .json files
        period_hours: Known period in hours (optional, skips estimation)
        fold: Fold number if using built-in checkpoints (0-4)
        checkpoint_path: Path to custom checkpoint (optional)

    Returns:
        List of AnalysisResult objects
    """
    # Find all CSV and JSON files
    csv_files = list(input_dir.glob("*.csv"))
    json_files = list(input_dir.glob("*.json"))
    input_files = sorted(csv_files + json_files)

    if not input_files:
        raise ValueError(f"No .csv or .json files found in {input_dir}")

    print(f"Found {len(input_files)} file(s) in {input_dir}")
    print("")

    results = []
    for i, input_path in enumerate(input_files, 1):
        print(f"[{i}/{len(input_files)}] Processing: {input_path.name}")
        try:
            result = run_single(
                input_path=input_path,
                period_hours=period_hours,
                fold=fold,
                checkpoint_path=checkpoint_path,
            )
            results.append(result)
            print(f"  ✓ Success: Period={result.period.period_hours:.2f}h, "
                  f"Pole=({result.best_pole.lambda_deg:.1f}°, {result.best_pole.beta_deg:.1f}°)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
        print("")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run pole prediction on asteroid lightcurve data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single asteroid with default checkpoint
  python run_pole_prediction.py --input asteroid_101.csv

  # Batch processing
  python run_pole_prediction.py --input-dir asteroids/ --output results.json

  # With known period (skip estimation)
  python run_pole_prediction.py --input asteroid_101.csv --period 8.5

  # Use different fold
  python run_pole_prediction.py --input asteroid_101.csv --fold 2

Input Formats:
  - DAMIT CSV: time_jd, brightness, sun_x, sun_y, sun_z, obs_x, obs_y, obs_z
  - DAMIT JSON: [{"object_id": "...", "points": "..."}]

Output Formats:
  - console: Pretty-printed results (default)
  - json: JSON file with all results
  - csv: CSV file with top predictions
        """
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input",
        type=Path,
        help="Input file (CSV or JSON format)"
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of input files (batch mode)"
    )

    # Optional parameters
    parser.add_argument(
        "--object-id",
        type=str,
        help="Object identifier (defaults to filename)"
    )
    parser.add_argument(
        "--period",
        type=float,
        help="Known period in hours (skip estimation if provided)"
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4],
        help="Fold number for built-in checkpoints (default: 0)"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to custom checkpoint file (experimental)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (JSON or CSV)"
    )
    parser.add_argument(
        "--format",
        choices=["console", "json", "csv"],
        default="console",
        help="Output format (default: console)"
    )

    args = parser.parse_args()

    # Validate inputs
    if args.input and not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    if args.input_dir and not args.input_dir.is_dir():
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)

    if args.checkpoint and not args.checkpoint.exists():
        print(f"ERROR: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # Run analysis
    try:
        if args.input:
            # Single file mode
            result = run_single(
                input_path=args.input,
                object_id=args.object_id,
                period_hours=args.period,
                fold=args.fold,
                checkpoint_path=args.checkpoint,
            )
            results = [result]
        else:
            # Batch mode
            results = run_batch(
                input_dir=args.input_dir,
                period_hours=args.period,
                fold=args.fold,
                checkpoint_path=args.checkpoint,
            )

        # Output results
        if args.format == "console":
            for result in results:
                print("")
                print(format_result_console(result))

        elif args.format == "json":
            output_data = {
                "results": [format_result_json(r) for r in results],
                "n_objects": len(results),
            }

            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(output_data, f, indent=2)
                print(f"Results saved to: {args.output}")
            else:
                print(json.dumps(output_data, indent=2))

        elif args.format == "csv":
            print("CSV format not yet implemented")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
