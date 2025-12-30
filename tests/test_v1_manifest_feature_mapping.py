#!/usr/bin/env python3
"""
Unit tests for V1 manifest feature mapping and downstream accessibility.

Tests:
- Manifest contains feature_names field with 28 elements
- feature_names match token_dim
- Feature indices can be retrieved by name (sun_u, obs_u, brightness, log_magerr)
- All expected features are present with correct semantics
- Downstream scripts can load and use feature_names
"""

import sys
import json
import pytest
import numpy as np
import tempfile
import os

sys.path.insert(0, "/mnt/d/Downloads/Colab Notebooks")

from scripts.build_pole_synth_multiepoch_v1_highgeo import (
    V1GenerationConfig, V1DatasetBuilder
)


class TestManifestStructure:
    """Test manifest structure and completeness."""

    def test_manifest_contains_feature_names(self):
        """Manifest should contain feature_names field."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            assert os.path.exists(manifest_path), "Manifest file not created"

            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            assert "feature_names" in manifest, "Manifest missing feature_names field"
            assert isinstance(manifest["feature_names"], list), "feature_names should be a list"

    def test_feature_names_count_matches_token_dim(self):
        """Number of feature_names should match token_dim."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            token_dim = manifest.get("token_dim", 28)
            feature_names = manifest["feature_names"]

            assert len(feature_names) == token_dim, \
                f"feature_names count {len(feature_names)} != token_dim {token_dim}"

    def test_feature_names_are_unique(self):
        """Feature names should be unique (no duplicates)."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            feature_names = manifest["feature_names"]
            unique_names = set(feature_names)

            assert len(feature_names) == len(unique_names), \
                f"Feature names contain duplicates: {[n for n in feature_names if feature_names.count(n) > 1]}"


class TestFeatureNameMapping:
    """Test feature name to index mapping."""

    def get_feature_index(self, feature_names, feature_name):
        """Utility to get feature index by name."""
        try:
            return feature_names.index(feature_name)
        except ValueError:
            return None

    def test_sun_u_features_exist(self):
        """Sun direction features should exist and be contiguous."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            feature_names = manifest["feature_names"]

            # sun_u should be at indices 0, 1, 2
            assert "sun_u_x" in feature_names, "sun_u_x not in feature_names"
            assert "sun_u_y" in feature_names, "sun_u_y not in feature_names"
            assert "sun_u_z" in feature_names, "sun_u_z not in feature_names"

            idx_x = self.get_feature_index(feature_names, "sun_u_x")
            idx_y = self.get_feature_index(feature_names, "sun_u_y")
            idx_z = self.get_feature_index(feature_names, "sun_u_z")

            assert idx_x == 0 and idx_y == 1 and idx_z == 2, \
                f"sun_u should be at [0:3], got x={idx_x}, y={idx_y}, z={idx_z}"

    def test_obs_u_features_exist(self):
        """Observer direction features should exist and be contiguous."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            feature_names = manifest["feature_names"]

            # obs_u should be at indices 3, 4, 5
            assert "obs_u_x" in feature_names, "obs_u_x not in feature_names"
            assert "obs_u_y" in feature_names, "obs_u_y not in feature_names"
            assert "obs_u_z" in feature_names, "obs_u_z not in feature_names"

            idx_x = self.get_feature_index(feature_names, "obs_u_x")
            idx_y = self.get_feature_index(feature_names, "obs_u_y")
            idx_z = self.get_feature_index(feature_names, "obs_u_z")

            assert idx_x == 3 and idx_y == 4 and idx_z == 5, \
                f"obs_u should be at [3:6], got x={idx_x}, y={idx_y}, z={idx_z}"

    def test_photometry_features_exist(self):
        """Photometry features should exist."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            feature_names = manifest["feature_names"]

            # brightness should be at index 6
            # log_magerr should be at index 7
            assert "brightness" in feature_names, "brightness not in feature_names"
            assert "log_magerr" in feature_names, "log_magerr not in feature_names"

            idx_brightness = self.get_feature_index(feature_names, "brightness")
            idx_magerr = self.get_feature_index(feature_names, "log_magerr")

            assert idx_brightness == 6, f"brightness should be at index 6, got {idx_brightness}"
            assert idx_magerr == 7, f"log_magerr should be at index 7, got {idx_magerr}"

    def test_padding_features_exist(self):
        """Padding features should fill remainder of token_dim."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            feature_names = manifest["feature_names"]
            token_dim = manifest["token_dim"]

            # Count padding features
            padding_features = [f for f in feature_names if f.startswith("pad_")]
            expected_pad_count = token_dim - 8  # 28 - 8 = 20 padding features

            assert len(padding_features) == expected_pad_count, \
                f"Expected {expected_pad_count} padding features, got {len(padding_features)}"


class TestDownstreamAccessibility:
    """Test that downstream scripts can load and use feature_names."""

    def test_load_manifest_from_disk(self):
        """Should be able to load manifest and access feature_names."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            builder.build_dataset()

            # Simulate downstream script loading manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            feature_names = manifest["feature_names"]

            # Should be able to build feature index map
            feature_index = {name: i for i, name in enumerate(feature_names)}

            # Verify key features are accessible
            assert feature_index["sun_u_x"] == 0
            assert feature_index["obs_u_x"] == 3
            assert feature_index["brightness"] == 6

    def test_extract_sun_u_by_name(self):
        """Downstream script should extract sun_u using feature_names."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            tokens, masks = builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            feature_names = manifest["feature_names"]

            # Build feature index
            feature_index = {name: i for i, name in enumerate(feature_names)}

            # Extract sun_u from sample
            sun_u_indices = [
                feature_index["sun_u_x"],
                feature_index["sun_u_y"],
                feature_index["sun_u_z"]
            ]

            sample_tokens = tokens[0, 0, :, :]  # First epoch of first object
            sun_u = sample_tokens[:, sun_u_indices]

            # Should be unit vectors
            norms = np.linalg.norm(sun_u, axis=1)
            assert np.allclose(norms, 1.0), \
                f"sun_u extracted by name should be unit vectors, got norms {norms[:5]}"

    def test_extract_obs_u_by_name(self):
        """Downstream script should extract obs_u using feature_names."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            tokens, masks = builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            feature_names = manifest["feature_names"]

            # Build feature index
            feature_index = {name: i for i, name in enumerate(feature_names)}

            # Extract obs_u from sample
            obs_u_indices = [
                feature_index["obs_u_x"],
                feature_index["obs_u_y"],
                feature_index["obs_u_z"]
            ]

            sample_tokens = tokens[0, 0, :, :]  # First epoch of first object
            obs_u = sample_tokens[:, obs_u_indices]

            # Should be unit vectors
            norms = np.linalg.norm(obs_u, axis=1)
            assert np.allclose(norms, 1.0), \
                f"obs_u extracted by name should be unit vectors, got norms {norms[:5]}"

    def test_extract_brightness_by_name(self):
        """Downstream script should extract brightness using feature_names."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            tokens, masks = builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            feature_names = manifest["feature_names"]

            # Build feature index
            feature_index = {name: i for i, name in enumerate(feature_names)}

            # Extract brightness from sample
            brightness_idx = feature_index["brightness"]

            sample_tokens = tokens[0, 0, :, :]  # First epoch of first object
            brightness = sample_tokens[:, brightness_idx]

            # Should have reasonable range (pole-dependent, so variable)
            assert brightness.min() >= -5.0, f"brightness too low: {brightness.min()}"
            assert brightness.max() <= 2.0, f"brightness too high: {brightness.max()}"


class TestConsistencyWithBuilder:
    """Test that feature_names in manifest match builder output structure."""

    def test_manifest_matches_token_output(self):
        """feature_names order should match actual token structure."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            tokens, masks = builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            feature_names = manifest["feature_names"]

            # Verify structure
            assert tokens.shape[-1] == len(feature_names), \
                f"Token dimension {tokens.shape[-1]} != feature_names count {len(feature_names)}"

            # Verify known structure: sun_u [0:3], obs_u [3:6], brightness [6], log_magerr [7]
            sample = tokens[0, 0, 0, :]  # Single token

            sun_u = sample[0:3]
            sun_norm = np.linalg.norm(sun_u)
            assert np.isclose(sun_norm, 1.0), f"sun_u at [0:3] should be unit vector, got norm {sun_norm}"

            obs_u = sample[3:6]
            obs_norm = np.linalg.norm(obs_u)
            assert np.isclose(obs_norm, 1.0), f"obs_u at [3:6] should be unit vector, got norm {obs_norm}"

            brightness = sample[6]
            assert -5.0 <= brightness <= 2.0, f"brightness at [6] out of range: {brightness}"

    def test_version_in_manifest(self):
        """Manifest should include version field for reproducibility."""
        config = V1GenerationConfig(n_objects=5, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            builder = V1DatasetBuilder(config, tmpdir)
            builder.build_dataset()

            # Load manifest
            manifest_path = os.path.join(tmpdir, "manifest.json")
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            assert "version" in manifest, "Manifest missing version field"
            assert manifest["version"] == "pole_synth_multiepoch_v1_highgeo", \
                f"Expected version 'pole_synth_multiepoch_v1_highgeo', got {manifest['version']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
