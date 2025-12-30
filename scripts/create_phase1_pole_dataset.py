"""
Create Phase 1 pole prediction dataset with 5-fold GroupKFold CV.

Dataset: Multi-apparition DAMIT asteroids
Features: 12D (geometry, photometry, rotational phase)
Splits: 5-fold GroupKFold (by asteroid ID)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
import argparse


def load_damit_asteroid(csv_path):
    """
    Load DAMIT asteroid CSV and extract metadata.

    Returns:
        data: DataFrame with all observations
        pole_true: (3,) true pole vector in Cartesian coords
        asteroid_id: Asteroid name
    """
    data = pd.read_csv(csv_path)
    asteroid_id = Path(csv_path).stem

    # Ground truth pole (ecliptic coordinates)
    lambda_deg = data['l'].iloc[0]  # Column 5
    beta_deg = data['b'].iloc[0]    # Column 6

    # Convert ecliptic to Cartesian (unit vector)
    lambda_rad = np.deg2rad(lambda_deg)
    beta_rad = np.deg2rad(beta_deg)

    pole_true = np.array([
        np.cos(beta_rad) * np.cos(lambda_rad),
        np.cos(beta_rad) * np.sin(lambda_rad),
        np.sin(beta_rad)
    ], dtype=np.float32)

    return data, pole_true, asteroid_id


def extract_features_12d(data, pole_true, period_pred=None, period_confidence=0.7):
    """
    Extract 12D features: geometry (6D) + photometry (2D) + phase (4D).

    CRITICAL: No per-object std normalization of magnitudes!

    Args:
        data: DataFrame with DAMIT observations
        pole_true: (3,) true pole
        period_pred: Predicted period in hours (None if unavailable)
        period_confidence: Confidence in predicted period

    Returns:
        apparitions: List of dicts with keys:
            'features': (N_obs, 12) array
            'apparition_id': Integer apparition ID
        metadata: Dict with asteroid info
    """
    # Split data into apparitions (gap > 30 days = new apparition)
    time_gaps = data['time'].diff()
    app_boundaries = np.where(time_gaps > 30)[0]  # +1 implicit in zipping
    app_starts = [0] + list(app_boundaries)
    app_ends = list(app_boundaries) + [len(data)]

    apparitions = []

    for app_idx, (start, end) in enumerate(zip(app_starts, app_ends)):
        app_data = data.iloc[start:end]

        # --- Geometry (6D) ---
        # Sun direction (columns 10-12)
        sun_xyz = app_data[['sun_ast_x', 'sun_ast_y', 'sun_ast_z']].values.astype(np.float32)
        sun_norm = np.linalg.norm(sun_xyz, axis=1, keepdims=True) + 1e-8
        sun_dir = sun_xyz / sun_norm  # (N_obs, 3)

        # Observer direction (columns 13-15)
        obs_xyz = app_data[['earth_ast_x', 'earth_ast_y', 'earth_ast_z']].values.astype(np.float32)
        obs_norm = np.linalg.norm(obs_xyz, axis=1, keepdims=True) + 1e-8
        obs_dir = obs_xyz / obs_norm  # (N_obs, 3)

        # --- Photometry (2D) ---
        # CRITICAL: mag_centered = mag - median(mag), NO std division!
        mag = app_data['relative_brightness'].values.astype(np.float32)
        mag_centered = mag - np.median(mag)  # Preserve amplitude scale

        # Phase angle (column 3)
        phase_angle = app_data['phase_angle'].values.astype(np.float32)
        phase_angle_rad = phase_angle * np.pi / 180.0

        # --- Rotational Phase (4D) ---
        if period_pred is not None and period_confidence > 0.7:
            # Phase at each observation
            time = app_data['time'].values.astype(np.float32)
            time_min = time[0]

            # Rotational phase: phi ∈ [0, 1)
            phi = np.fmod((time - time_min) / (period_pred / 24.0), 1.0)

            # Fourier features
            sin_phi = np.sin(2 * np.pi * phi)
            cos_phi = np.cos(2 * np.pi * phi)
            sin_2phi = np.sin(4 * np.pi * phi)
            cos_2phi = np.cos(4 * np.pi * phi)
        else:
            # Fallback: zeros (model will struggle without phase)
            n_obs = len(app_data)
            sin_phi = np.zeros(n_obs, dtype=np.float32)
            cos_phi = np.zeros(n_obs, dtype=np.float32)
            sin_2phi = np.zeros(n_obs, dtype=np.float32)
            cos_2phi = np.zeros(n_obs, dtype=np.float32)

        # Concatenate: (N_obs, 12)
        features = np.column_stack([
            sun_dir,  # 6D (columns 0-2, 3-5 for components? no, 0-2)
            obs_dir,  # 6D
            mag_centered[:, np.newaxis],  # 1D
            phase_angle_rad[:, np.newaxis],  # 1D
            sin_phi[:, np.newaxis],  # 1D
            cos_phi[:, np.newaxis],  # 1D
            sin_2phi[:, np.newaxis],  # 1D
            cos_2phi[:, np.newaxis]  # 1D
        ]).astype(np.float32)

        # Verify shape
        assert features.shape[1] == 12, f"Expected 12 features, got {features.shape[1]}"

        apparitions.append({
            'features': features,
            'apparition_id': app_idx,
            'n_obs': len(app_data)
        })

    metadata = {
        'pole_true': pole_true,
        'n_apparitions': len(apparitions),
        'total_obs': len(data),
        'period_pred': period_pred,
        'timespan_days': data['time'].max() - data['time'].min()
    }

    return apparitions, metadata


def create_phase1_dataset(
    damit_dir: Path,
    output_dir: Path,
    n_folds: int = 5,
    random_seed: int = 42
):
    """
    Create Phase 1 dataset with 5-fold GroupKFold splitting.

    Args:
        damit_dir: Path to DAMIT_csv_high directory
        output_dir: Path to save dataset
        n_folds: Number of CV folds
        random_seed: Random seed for reproducibility
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all asteroids
    print("Loading DAMIT asteroids...")
    csv_files = sorted(damit_dir.glob('*.csv'))
    print(f"Found {len(csv_files)} asteroids")

    asteroids = []
    for csv_path in csv_files:
        try:
            data, pole_true, asteroid_id = load_damit_asteroid(csv_path)

            # Require ≥ 2 apparitions for Phase 1 (but all DAMIT has this)
            n_apps = 1 + sum(data['time'].diff() > 30)
            if n_apps < 2:
                continue

            apparitions, metadata = extract_features_12d(data, pole_true)

            asteroids.append({
                'id': asteroid_id,
                'apparitions': apparitions,
                'metadata': metadata
            })
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            continue

    print(f"Loaded {len(asteroids)} asteroids with ≥2 apparitions")

    # Create 5-fold GroupKFold splits
    np.random.seed(random_seed)
    asteroid_ids = np.array([a['id'] for a in asteroids])
    groups = np.arange(len(asteroids))  # Group by asteroid index

    gkf = GroupKFold(n_splits=n_folds)
    splits = list(gkf.split(asteroid_ids, groups=groups))

    # Save each fold
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"\nFold {fold_idx}:")

        train_asteroids = [asteroids[i] for i in train_idx]
        test_asteroids = [asteroids[i] for i in test_idx]

        print(f"  Train: {len(train_asteroids)} asteroids")
        print(f"  Test: {len(test_asteroids)} asteroids")

        # Save fold data
        fold_dir = output_dir / f'fold_{fold_idx}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        # Save as NPZ for efficient loading
        train_data = {
            'asteroids': train_asteroids,
            'indices': train_idx
        }
        test_data = {
            'asteroids': test_asteroids,
            'indices': test_idx
        }

        # Save metadata as JSON (can't pickle apparitions directly)
        train_meta = {
            'asteroid_ids': [a['id'] for a in train_asteroids],
            'n_asteroids': len(train_asteroids),
            'fold_idx': fold_idx
        }
        test_meta = {
            'asteroid_ids': [a['id'] for a in test_asteroids],
            'n_asteroids': len(test_asteroids),
            'fold_idx': fold_idx
        }

        with open(fold_dir / 'train_meta.json', 'w') as f:
            json.dump(train_meta, f, indent=2)

        with open(fold_dir / 'test_meta.json', 'w') as f:
            json.dump(test_meta, f, indent=2)

        # Save pickle of asteroid data
        import pickle
        with open(fold_dir / 'train_asteroids.pkl', 'wb') as f:
            pickle.dump(train_asteroids, f)

        with open(fold_dir / 'test_asteroids.pkl', 'wb') as f:
            pickle.dump(test_asteroids, f)

        print(f"  Saved to {fold_dir}")

    # Save overall metadata
    overall_meta = {
        'n_asteroids': len(asteroids),
        'n_folds': n_folds,
        'asteroid_ids': [a['id'] for a in asteroids],
        'random_seed': random_seed
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(overall_meta, f, indent=2)

    print(f"\nPhase 1 dataset created at {output_dir}")
    print(f"Total: {len(asteroids)} asteroids × 5 folds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Phase 1 pole prediction dataset')
    parser.add_argument('--damit_dir', type=Path,
                       default=Path('/mnt/d/Downloads/Colab Notebooks/DAMIT_csv_high'),
                       help='Path to DAMIT_csv_high directory')
    parser.add_argument('--output_dir', type=Path,
                       default=Path('/tmp/phase1_pole_dataset'),
                       help='Output directory for dataset')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    create_phase1_dataset(
        damit_dir=args.damit_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        random_seed=args.seed
    )
