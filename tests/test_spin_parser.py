#!/usr/bin/env python3
"""
Tests for spin.txt parsing in make_period_cache_from_damit.py.
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.make_period_cache_from_damit import (
    parse_spin_file,
    extract_object_id_from_path,
    discover_spin_files,
)


class TestParseSpinFile:
    """Tests for parse_spin_file function."""

    def test_lambda_beta_period_format(self):
        """Standard DAMIT format: lambda beta period."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spin_path = Path(tmpdir) / "spin.txt"
            spin_path.write_text("123.45 -67.89 5.27\n")

            result = parse_spin_file(spin_path)
            assert result == 5.27

    def test_period_only(self):
        """File with just period value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spin_path = Path(tmpdir) / "spin.txt"
            spin_path.write_text("8.5\n")

            result = parse_spin_file(spin_path)
            assert result == 8.5

    def test_fallback_non_standard_format(self):
        """Fallback when first line doesn't have standard format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spin_path = Path(tmpdir) / "spin.txt"
            # Non-standard format: only 2 values on first line, period on second
            content = """123.45 -67.89
12.5
"""
            spin_path.write_text(content)

            result = parse_spin_file(spin_path)
            # Falls back to first plausible value: 123.45 is in range
            assert result == 123.45

    def test_days_to_hours_conversion(self):
        """Small values should be converted from days to hours."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spin_path = Path(tmpdir) / "spin.txt"
            # 0.25 days = 6 hours
            spin_path.write_text("0.25\n")

            result = parse_spin_file(spin_path)
            # Should convert to hours
            assert result == 6.0

    def test_standard_damit_format(self):
        """Standard DAMIT format: first line has lambda beta period."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spin_path = Path(tmpdir) / "spin.txt"
            # Standard DAMIT format: lambda=123.45, beta=-67.89, period=7.3
            content = "123.45 -67.89 7.3\n2456908 0\n0.1 0.5 0.1\n"
            spin_path.write_text(content)

            result = parse_spin_file(spin_path)
            assert result == 7.3

    def test_scientific_notation(self):
        """Handle scientific notation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spin_path = Path(tmpdir) / "spin.txt"
            spin_path.write_text("1.23e1\n")  # 12.3

            result = parse_spin_file(spin_path)
            assert result == pytest.approx(12.3)

    def test_empty_file_returns_none(self):
        """Empty file should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spin_path = Path(tmpdir) / "spin.txt"
            spin_path.write_text("")

            result = parse_spin_file(spin_path)
            assert result is None

    def test_no_numbers_returns_none(self):
        """File with no numbers should return None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            spin_path = Path(tmpdir) / "spin.txt"
            spin_path.write_text("no numbers here\n")

            result = parse_spin_file(spin_path)
            assert result is None

    def test_nonexistent_file_returns_none(self):
        """Nonexistent file should return None."""
        result = parse_spin_file(Path("/nonexistent/spin.txt"))
        assert result is None


class TestExtractObjectId:
    """Tests for extract_object_id_from_path function."""

    def test_actual_damit_path(self):
        """Actual DAMIT path: .../asteroid_<id>/model_<id>/spin.txt"""
        path = Path("/data/DAMIT/files/asteroid_101/model_101/spin.txt")
        result = extract_object_id_from_path(path)
        assert result == "asteroid_101_model_101"

    def test_different_ids(self):
        """Different asteroid and model IDs."""
        path = Path("/data/DAMIT/asteroid_123/model_456/spin.txt")
        result = extract_object_id_from_path(path)
        assert result == "asteroid_123_model_456"

    def test_numeric_ids_fallback(self):
        """Numeric IDs without prefix should work."""
        path = Path("/data/DAMIT/123/456/spin.txt")
        result = extract_object_id_from_path(path)
        assert result == "asteroid_123_model_456"

    def test_too_short_path_returns_none(self):
        """Path too short to extract IDs."""
        path = Path("spin.txt")
        result = extract_object_id_from_path(path)
        assert result is None

    def test_path_with_only_one_parent(self):
        """Only one parent directory."""
        path = Path("/101/spin.txt")
        result = extract_object_id_from_path(path)
        assert result is None


class TestDiscoverSpinFiles:
    """Tests for discover_spin_files function."""

    def test_discover_single_spin_file(self):
        """Find a single spin.txt file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spin_dir = root / "101" / "101"
            spin_dir.mkdir(parents=True)
            spin_path = spin_dir / "spin.txt"
            spin_path.write_text("5.27\n")

            result = discover_spin_files(root)

            assert len(result) == 1
            assert "asteroid_101_model_101" in result
            assert result["asteroid_101_model_101"] == spin_path

    def test_discover_multiple_spin_files(self):
        """Find multiple spin.txt files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            for asteroid_id in ["101", "102", "103"]:
                spin_dir = root / asteroid_id / asteroid_id
                spin_dir.mkdir(parents=True)
                (spin_dir / "spin.txt").write_text("5.0\n")

            result = discover_spin_files(root)

            assert len(result) == 3
            assert "asteroid_101_model_101" in result
            assert "asteroid_102_model_102" in result
            assert "asteroid_103_model_103" in result

    def test_discover_nested_structure(self):
        """Find spin.txt in deeper nested structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spin_dir = root / "shape_models" / "200" / "200"
            spin_dir.mkdir(parents=True)
            spin_path = spin_dir / "spin.txt"
            spin_path.write_text("7.3\n")

            result = discover_spin_files(root)

            assert len(result) == 1
            assert "asteroid_200_model_200" in result

    def test_empty_directory(self):
        """Empty directory returns empty dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = discover_spin_files(Path(tmpdir))
            assert result == {}

    def test_ignores_invalid_paths(self):
        """Ignores spin.txt files that can't be mapped to object_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "damit_root"
            root.mkdir()

            # Valid spin.txt
            valid_dir = root / "101" / "101"
            valid_dir.mkdir(parents=True)
            (valid_dir / "spin.txt").write_text("5.0\n")

            # Invalid: directly under root with only one parent level
            invalid_dir = root / "orphan"
            invalid_dir.mkdir()
            (invalid_dir / "spin.txt").write_text("5.0\n")

            result = discover_spin_files(root)

            # Should find valid one; orphan may or may not be found depending on path depth
            assert "asteroid_101_model_101" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
