#!/usr/bin/env python3
"""
Tests for period cache I/O operations.
"""

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pole_synth.period_cache import (
    PeriodInfo,
    load_period_cache,
    save_period_cache,
    get_period_hours,
    validate_coverage,
    require_coverage,
)


class TestPeriodInfo:
    """Tests for PeriodInfo dataclass."""

    def test_create_period_info(self):
        info = PeriodInfo(period_hours=5.27)
        assert info.period_hours == 5.27

    def test_to_dict(self):
        info = PeriodInfo(period_hours=12.5)
        d = info.to_dict()
        assert d == {"period_hours": 12.5}

    def test_from_dict(self):
        d = {"period_hours": 7.3}
        info = PeriodInfo.from_dict(d)
        assert info.period_hours == 7.3

    def test_roundtrip_dict(self):
        original = PeriodInfo(period_hours=3.14159)
        d = original.to_dict()
        restored = PeriodInfo.from_dict(d)
        assert restored.period_hours == original.period_hours


class TestSaveLoadCache:
    """Tests for save/load operations."""

    def test_roundtrip_save_load(self):
        """Save and load should preserve data."""
        mapping = {
            "damit_101_101": PeriodInfo(period_hours=5.27),
            "damit_102_102": PeriodInfo(period_hours=12.5),
            "damit_200_200": PeriodInfo(period_hours=3.14),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            save_period_cache(path, mapping)

            loaded = load_period_cache(path)

            assert len(loaded) == len(mapping)
            for oid, info in mapping.items():
                assert oid in loaded
                assert loaded[oid].period_hours == info.period_hours

    def test_save_creates_parent_dirs(self):
        """Save should create parent directories if needed."""
        mapping = {"damit_101_101": PeriodInfo(period_hours=5.0)}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "cache.json"
            save_period_cache(path, mapping)

            assert path.exists()
            loaded = load_period_cache(path)
            assert len(loaded) == 1

    def test_load_nonexistent_raises(self):
        """Load should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_period_cache("/nonexistent/path/cache.json")

    def test_load_invalid_json_raises(self):
        """Load should raise for invalid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.json"
            path.write_text("not valid json {{{")

            with pytest.raises(json.JSONDecodeError):
                load_period_cache(path)

    def test_json_format_is_readable(self):
        """Saved JSON should be human-readable."""
        mapping = {
            "damit_101_101": PeriodInfo(period_hours=5.27),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            save_period_cache(path, mapping)

            content = path.read_text()
            data = json.loads(content)

            assert "damit_101_101" in data
            assert data["damit_101_101"]["period_hours"] == 5.27


class TestGetPeriodHours:
    """Tests for get_period_hours function."""

    def test_get_existing(self):
        mapping = {
            "damit_101_101": PeriodInfo(period_hours=5.27),
            "damit_102_102": PeriodInfo(period_hours=12.5),
        }

        result = get_period_hours(mapping, "damit_101_101")
        assert result == 5.27

        result = get_period_hours(mapping, "damit_102_102")
        assert result == 12.5

    def test_get_nonexistent_returns_none(self):
        mapping = {
            "damit_101_101": PeriodInfo(period_hours=5.27),
        }

        result = get_period_hours(mapping, "damit_999_999")
        assert result is None


class TestValidateCoverage:
    """Tests for validate_coverage function."""

    def test_full_coverage(self):
        mapping = {
            "a": PeriodInfo(period_hours=1.0),
            "b": PeriodInfo(period_hours=2.0),
            "c": PeriodInfo(period_hours=3.0),
        }
        object_ids = ["a", "b", "c"]

        loaded, total, frac = validate_coverage(mapping, object_ids)

        assert loaded == 3
        assert total == 3
        assert frac == 1.0

    def test_partial_coverage(self):
        mapping = {
            "a": PeriodInfo(period_hours=1.0),
            "b": PeriodInfo(period_hours=2.0),
        }
        object_ids = ["a", "b", "c", "d"]

        loaded, total, frac = validate_coverage(mapping, object_ids)

        assert loaded == 2
        assert total == 4
        assert frac == 0.5

    def test_no_coverage(self):
        mapping = {}
        object_ids = ["a", "b", "c"]

        loaded, total, frac = validate_coverage(mapping, object_ids)

        assert loaded == 0
        assert total == 3
        assert frac == 0.0

    def test_empty_object_ids(self):
        mapping = {"a": PeriodInfo(period_hours=1.0)}
        object_ids = []

        loaded, total, frac = validate_coverage(mapping, object_ids)

        assert loaded == 0
        assert total == 0
        assert frac == 1.0  # Edge case: empty is considered full coverage


class TestRequireCoverage:
    """Tests for require_coverage function."""

    def test_passes_when_sufficient(self):
        mapping = {
            "a": PeriodInfo(period_hours=1.0),
            "b": PeriodInfo(period_hours=2.0),
        }
        object_ids = ["a", "b"]

        loaded, total, frac = require_coverage(mapping, object_ids, min_frac=0.95)

        assert loaded == 2
        assert total == 2
        assert frac == 1.0

    def test_raises_when_insufficient(self):
        mapping = {
            "a": PeriodInfo(period_hours=1.0),
        }
        object_ids = ["a", "b", "c", "d"]

        with pytest.raises(ValueError) as excinfo:
            require_coverage(mapping, object_ids, min_frac=0.95)

        assert "Insufficient period coverage" in str(excinfo.value)
        assert "1/4" in str(excinfo.value)

    def test_context_in_error_message(self):
        mapping = {}
        object_ids = ["a"]

        with pytest.raises(ValueError) as excinfo:
            require_coverage(mapping, object_ids, min_frac=0.5, context="test dataset")

        assert "test dataset" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
