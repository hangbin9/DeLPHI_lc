"""
Phase 29I: Smoke tests for reporting module.

Tests that:
- reporting.py generates markdown with required sections
- JSON schema has required keys (version, folds, aggregates, commands)
- Aggregation works correctly on synthetic fold results
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from pole_synth.reporting import (
    aggregate_fold_results,
    build_paper_report,
    build_paper_results_json,
    compute_mean_std,
    format_by_label_table,
    format_calibration_table,
    format_diagnostics_section,
    format_mean_std,
    format_oracle_gap_table,
    format_overall_metrics_table,
    save_paper_results,
)
from pole_synth.version import __version__, get_version, get_version_info


# ============================================================================
# Fixtures: Synthetic fold results
# ============================================================================


def make_synthetic_fold_result(seed: int = 0) -> Dict[str, Any]:
    """Create synthetic fold result with realistic structure."""
    rng = np.random.default_rng(seed)

    # Base errors with some noise
    oracle_base = 5.0 + rng.uniform(-0.5, 0.5)
    learned_base = 6.0 + rng.uniform(-0.5, 0.5)
    agreement_base = 6.5 + rng.uniform(-0.5, 0.5)
    naive_base = 8.0 + rng.uniform(-0.5, 0.5)

    def make_metrics(base: float) -> Dict[str, Any]:
        noise = rng.uniform(-0.3, 0.3, 4)
        return {
            'overall': {
                'set_err_mean': base + noise[0],
                'set_err_median': base - 0.2 + noise[1],
                'can_err_mean': base * 0.8 + noise[2],
                'can_err_median': base * 0.8 - 0.2 + noise[3],
            },
            'by_label': {
                'unique': {
                    'set_err_mean': base - 1.0 + rng.uniform(-0.2, 0.2),
                    'set_err_median': base - 1.2 + rng.uniform(-0.2, 0.2),
                    'can_err_mean': base * 0.7 + rng.uniform(-0.2, 0.2),
                    'can_err_median': base * 0.7 - 0.2 + rng.uniform(-0.2, 0.2),
                },
                'multi_close': {
                    'set_err_mean': base + 0.5 + rng.uniform(-0.2, 0.2),
                    'set_err_median': base + 0.3 + rng.uniform(-0.2, 0.2),
                    'can_err_mean': base * 0.85 + rng.uniform(-0.2, 0.2),
                    'can_err_median': base * 0.85 - 0.2 + rng.uniform(-0.2, 0.2),
                },
                'multi_far': {
                    'set_err_mean': base + 2.0 + rng.uniform(-0.2, 0.2),
                    'set_err_median': base + 1.8 + rng.uniform(-0.2, 0.2),
                    'can_err_mean': base * 0.9 + rng.uniform(-0.2, 0.2),
                    'can_err_median': base * 0.9 - 0.2 + rng.uniform(-0.2, 0.2),
                },
            },
        }

    result = {
        'oracle_k2': make_metrics(oracle_base),
        'selected_learned_k2': {
            **make_metrics(learned_base),
            'selector_accuracy': 0.72 + rng.uniform(-0.05, 0.05),
        },
        'selected_agreement_k2': {
            **make_metrics(agreement_base),
            'selector_accuracy': 0.68 + rng.uniform(-0.05, 0.05),
        },
        'naive0_k2': make_metrics(naive_base),
        'calibration': {
            'bins': [
                {'bin_idx': i, 'gap_mean': 3.0 - i * 0.5 + rng.uniform(-0.1, 0.1),
                 'accuracy': 0.5 + i * 0.1 + rng.uniform(-0.02, 0.02),
                 'n_objects': int(100 + rng.integers(-20, 20))}
                for i in range(5)
            ],
        },
        'diagnostics': {
            'collapse_rate': 0.08 + rng.uniform(-0.02, 0.02),
            'complementarity_rate': 0.35 + rng.uniform(-0.05, 0.05),
            'mean_interhyp_angle': 25.0 + rng.uniform(-3, 3),
        },
    }

    return result


@pytest.fixture
def synthetic_fold_results() -> Dict[str, Dict[str, Any]]:
    """Create 3 synthetic folds for testing."""
    return {
        'fold_0': make_synthetic_fold_result(seed=0),
        'fold_1': make_synthetic_fold_result(seed=1),
        'fold_2': make_synthetic_fold_result(seed=2),
    }


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Sample config for testing."""
    return {
        'seed': 42,
        'device': 'cpu',
        'folds': [0, 1, 2],
        'selector_mode': 'learned',
        'label_mode': 'hybrid',
    }


@pytest.fixture
def sample_commands() -> list:
    """Sample commands for testing."""
    return [
        'python scripts/phase29i_paper_run.py --csv-dir data --folds 0 1 2',
    ]


# ============================================================================
# Tests: Version module
# ============================================================================


class TestVersionModule:
    """Tests for version.py module."""

    def test_version_string_format(self):
        """Version should be in expected format."""
        assert __version__ == "29I-rc1"

    def test_get_version_returns_string(self):
        """get_version() should return version string."""
        version = get_version()
        assert isinstance(version, str)
        assert version == __version__

    def test_get_version_info_structure(self):
        """get_version_info() should return dict with required keys."""
        info = get_version_info()
        assert isinstance(info, dict)
        assert 'version' in info
        assert 'phase' in info
        assert 'release_type' in info
        assert 'features' in info
        assert info['phase'] == '29I'
        assert info['release_type'] == 'rc1'
        assert isinstance(info['features'], list)


# ============================================================================
# Tests: Utility functions
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions in reporting.py."""

    def test_compute_mean_std_empty(self):
        """Empty list should return zeros."""
        mean, std = compute_mean_std([])
        assert mean == 0.0
        assert std == 0.0

    def test_compute_mean_std_single(self):
        """Single value should have zero std."""
        mean, std = compute_mean_std([5.0])
        assert mean == 5.0
        assert std == 0.0

    def test_compute_mean_std_multiple(self):
        """Multiple values should compute correctly."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        mean, std = compute_mean_std(values)
        assert mean == 3.0
        assert abs(std - np.std(values)) < 1e-10

    def test_format_mean_std_default_precision(self):
        """Default precision should be 2 decimal places."""
        result = format_mean_std(5.123, 1.456)
        assert result == "5.12±1.46"

    def test_format_mean_std_custom_precision(self):
        """Custom precision should work."""
        result = format_mean_std(5.1234, 1.4567, precision=3)
        assert result == "5.123±1.457"


# ============================================================================
# Tests: Aggregation
# ============================================================================


class TestAggregation:
    """Tests for aggregate_fold_results function."""

    def test_aggregate_empty_results(self):
        """Empty fold results should return empty dict."""
        result = aggregate_fold_results({})
        assert result == {}

    def test_aggregate_has_required_keys(self, synthetic_fold_results):
        """Aggregated results should have all required method keys."""
        agg = aggregate_fold_results(synthetic_fold_results)

        # Check main method groups exist
        assert 'oracle_k2' in agg
        assert 'selected_learned_k2' in agg
        assert 'selected_agreement_k2' in agg
        assert 'naive0_k2' in agg

        # Check overall metrics exist
        for method in ['oracle_k2', 'selected_learned_k2', 'selected_agreement_k2', 'naive0_k2']:
            assert 'overall' in agg[method]
            assert 'by_label' in agg[method]
            assert 'set_err_mean_mean' in agg[method]['overall']
            assert 'set_err_mean_std' in agg[method]['overall']

    def test_aggregate_has_by_label_breakdown(self, synthetic_fold_results):
        """Aggregated results should have by-label breakdowns."""
        agg = aggregate_fold_results(synthetic_fold_results)

        for method in ['oracle_k2', 'selected_learned_k2']:
            by_label = agg[method]['by_label']
            assert 'unique' in by_label
            assert 'multi_close' in by_label
            assert 'multi_far' in by_label

    def test_aggregate_has_oracle_gap(self, synthetic_fold_results):
        """Aggregated results should have oracle gap analysis."""
        agg = aggregate_fold_results(synthetic_fold_results)

        assert 'oracle_gap_learned' in agg
        assert 'oracle_gap_agreement' in agg
        assert 'mean' in agg['oracle_gap_learned']
        assert 'std' in agg['oracle_gap_learned']

    def test_aggregate_has_calibration(self, synthetic_fold_results):
        """Aggregated results should have calibration data."""
        agg = aggregate_fold_results(synthetic_fold_results)

        assert 'calibration' in agg
        assert isinstance(agg['calibration'], list)
        assert len(agg['calibration']) == 5  # 5 bins

        for bin_data in agg['calibration']:
            assert 'bin_idx' in bin_data
            assert 'gap_mean' in bin_data
            assert 'gap_std' in bin_data
            assert 'accuracy_mean' in bin_data
            assert 'total_count' in bin_data

    def test_aggregate_has_diagnostics(self, synthetic_fold_results):
        """Aggregated results should have diagnostics."""
        agg = aggregate_fold_results(synthetic_fold_results)

        assert 'diagnostics' in agg
        diag = agg['diagnostics']
        assert 'collapse_rate_mean' in diag
        assert 'collapse_rate_std' in diag
        assert 'complementarity_rate_mean' in diag
        assert 'mean_interhyp_angle_mean' in diag

    def test_aggregate_has_selector_accuracy(self, synthetic_fold_results):
        """Aggregated results should have selector accuracy."""
        agg = aggregate_fold_results(synthetic_fold_results)

        assert 'selector_accuracy_mean' in agg['selected_learned_k2']
        assert 'selector_accuracy_std' in agg['selected_learned_k2']
        assert 'selector_accuracy_mean' in agg['selected_agreement_k2']


# ============================================================================
# Tests: JSON structure
# ============================================================================


class TestJSONStructure:
    """Tests for build_paper_results_json function."""

    def test_json_has_required_keys(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """JSON should have all required top-level keys."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )

        assert 'version' in results
        assert 'version_info' in results
        assert 'timestamp' in results
        assert 'config' in results
        assert 'commands' in results
        assert 'n_folds' in results
        assert 'folds' in results
        assert 'aggregates' in results

    def test_json_version_matches(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """JSON version should match __version__."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        assert results['version'] == __version__

    def test_json_fold_count_correct(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """JSON n_folds should match input."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        assert results['n_folds'] == 3

    def test_json_serializable(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Results should be JSON serializable."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        # Should not raise
        json_str = json.dumps(results, default=str)
        assert len(json_str) > 0

        # Should be parseable
        parsed = json.loads(json_str)
        assert parsed['version'] == results['version']


# ============================================================================
# Tests: Markdown report
# ============================================================================


class TestMarkdownReport:
    """Tests for markdown report generation."""

    def test_report_has_title(self, synthetic_fold_results, sample_commands, sample_config):
        """Report should have title."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        report = build_paper_report(results)

        assert '# Pole Prediction Results (Phase 29I)' in report

    def test_report_has_version_info(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Report should have version information."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        report = build_paper_report(results)

        assert f'**Version:** {__version__}' in report
        assert '**Generated:**' in report
        assert '**Folds:** 3' in report

    def test_report_has_overall_results_table(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Report should have overall results table."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        report = build_paper_report(results)

        assert '## Overall Results' in report
        assert '| Method | Set Error (Mean) | Set Error (Median) | Can Error (Mean) |' in report
        assert 'Oracle@2' in report
        assert 'Selected@2 (Learned)' in report
        assert 'Selected@2 (Agreement)' in report
        assert 'Naive0@2' in report

    def test_report_has_by_label_breakdown(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Report should have by-label breakdown."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        report = build_paper_report(results)

        assert '## By-Label Breakdown' in report
        assert '### Oracle@2 by Label' in report
        assert 'unique' in report
        assert 'multi_close' in report
        assert 'multi_far' in report

    def test_report_has_oracle_gap(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Report should have oracle gap analysis."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        report = build_paper_report(results)

        assert '## Oracle Gap Analysis' in report
        assert 'Gap = Selected - Oracle' in report

    def test_report_has_calibration(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Report should have calibration section."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        report = build_paper_report(results)

        assert '## Selector Calibration' in report
        assert 'Confidence bins' in report
        assert '**Calibrated (monotonic):**' in report

    def test_report_has_diagnostics(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Report should have diagnostics section."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        report = build_paper_report(results)

        assert '## Diagnostics' in report
        assert 'Collapse Rate' in report
        assert 'Complementarity Rate' in report
        assert 'Mean Inter-Hypothesis Angle' in report
        assert 'Selector Accuracy' in report

    def test_report_has_commands(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Report should have commands section."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        report = build_paper_report(results)

        assert '## Commands Executed' in report
        assert '```bash' in report
        assert sample_commands[0] in report


# ============================================================================
# Tests: Formatting functions
# ============================================================================


class TestFormattingFunctions:
    """Tests for individual formatting functions."""

    def test_format_overall_metrics_table(self, synthetic_fold_results):
        """Overall metrics table should format correctly."""
        agg = aggregate_fold_results(synthetic_fold_results)
        table = format_overall_metrics_table(agg)

        assert '## Overall Results' in table
        assert '| Method |' in table
        assert 'Oracle@2' in table
        assert '°' in table  # Degree symbol

    def test_format_by_label_table(self, synthetic_fold_results):
        """By-label table should format correctly."""
        agg = aggregate_fold_results(synthetic_fold_results)
        table = format_by_label_table(agg, 'oracle_k2', 'Oracle@2')

        assert '### Oracle@2 by Label' in table
        assert '| Label |' in table
        assert 'unique' in table

    def test_format_oracle_gap_table(self, synthetic_fold_results):
        """Oracle gap table should format correctly."""
        agg = aggregate_fold_results(synthetic_fold_results)
        table = format_oracle_gap_table(agg)

        assert '## Oracle Gap Analysis' in table
        assert 'Learned' in table
        assert 'Agreement' in table

    def test_format_calibration_table(self, synthetic_fold_results):
        """Calibration table should format correctly."""
        agg = aggregate_fold_results(synthetic_fold_results)
        table = format_calibration_table(agg)

        assert '## Selector Calibration' in table
        assert '| Bin |' in table
        assert 'monotonic' in table.lower()

    def test_format_calibration_empty(self):
        """Empty calibration should show appropriate message."""
        table = format_calibration_table({})
        assert 'No calibration data available' in table

    def test_format_diagnostics_section(self, synthetic_fold_results):
        """Diagnostics section should format correctly."""
        agg = aggregate_fold_results(synthetic_fold_results)
        section = format_diagnostics_section(agg)

        assert '## Diagnostics' in section
        assert 'Collapse Rate' in section
        assert '%' in section  # Percentage format


# ============================================================================
# Tests: File saving
# ============================================================================


class TestFileSaving:
    """Tests for save_paper_results function."""

    def test_save_creates_files(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """save_paper_results should create both JSON and MD files."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / 'results.json'
            md_path = Path(tmpdir) / 'report.md'

            save_paper_results(results, json_path, md_path)

            assert json_path.exists()
            assert md_path.exists()

    def test_saved_json_valid(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Saved JSON should be valid and contain expected data."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / 'results.json'
            md_path = Path(tmpdir) / 'report.md'

            save_paper_results(results, json_path, md_path)

            with open(json_path) as f:
                loaded = json.load(f)

            assert loaded['version'] == __version__
            assert loaded['n_folds'] == 3

    def test_saved_markdown_valid(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Saved markdown should contain expected content."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / 'results.json'
            md_path = Path(tmpdir) / 'report.md'

            save_paper_results(results, json_path, md_path)

            content = md_path.read_text()

            assert '# Pole Prediction Results' in content
            assert '## Overall Results' in content

    def test_save_creates_parent_dirs(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """save_paper_results should create parent directories."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / 'nested' / 'path' / 'results.json'
            md_path = Path(tmpdir) / 'nested' / 'path' / 'report.md'

            save_paper_results(results, json_path, md_path)

            assert json_path.exists()
            assert md_path.exists()


# ============================================================================
# Tests: Stability / Determinism
# ============================================================================


class TestDeterminism:
    """Tests for deterministic output."""

    def test_aggregation_deterministic(self, synthetic_fold_results):
        """Same input should produce same aggregation."""
        agg1 = aggregate_fold_results(synthetic_fold_results)
        agg2 = aggregate_fold_results(synthetic_fold_results)

        # Compare JSON representations for equality
        assert json.dumps(agg1, sort_keys=True) == json.dumps(agg2, sort_keys=True)

    def test_report_structure_stable(
        self, synthetic_fold_results, sample_commands, sample_config
    ):
        """Report sections should appear in stable order."""
        results = build_paper_results_json(
            synthetic_fold_results, sample_commands, sample_config
        )
        report = build_paper_report(results)

        # Find positions of section headers
        sections = [
            '# Pole Prediction Results',
            '## Overall Results',
            '## By-Label Breakdown',
            '## Oracle Gap Analysis',
            '## Selector Calibration',
            '## Diagnostics',
            '## Commands Executed',
        ]

        positions = [report.find(s) for s in sections]

        # All should be found
        assert all(p >= 0 for p in positions), "Not all sections found"

        # Should be in ascending order
        assert positions == sorted(positions), "Sections not in expected order"
