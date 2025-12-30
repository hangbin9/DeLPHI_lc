# Release Notes: Phase 29I - Release Candidate 1

**Version:** 29I-rc1
**Date:** December 2024
**Status:** Release Candidate

## Overview

Phase 29I delivers the final packaging and reproducible "paper run" capability for the multi-hypothesis pole prediction system. This release candidate provides a one-command script that runs the complete K=2 multi-hypothesis pipeline with learned selector, producing paper-ready results.

## Key Features

### 1. One-Command Paper Run (`scripts/phase29i_paper_run.py`)

Run the complete evaluation pipeline with a single command:

```bash
python scripts/phase29i_paper_run.py \
    --csv-dir /path/to/csv \
    --artifacts-root /path/to/artifacts \
    --selector-root /path/to/selectors \
    --folds 0 1 2 \
    --outdir results/paper_run
```

**Capabilities:**
- Automatic model and selector loading per fold
- Optional selector training with `--train-selector`
- Both learned and heuristic (agreement) selector modes
- Three label modes: `hybrid` (default), `canonical`, `set`
- Deterministic execution with `--seed` control

### 2. Unified Report Builder (`pole_synth/reporting.py`)

Generates paper-ready outputs:
- **JSON artifact** (`paper_results.json`): Complete structured results with per-fold and aggregated metrics
- **Markdown report** (`paper_report.md`): Formatted tables and analysis ready for paper

**Report Contents:**
- Overall results table (Oracle@2, Selected@2, Naive0@2)
- By-label breakdown (unique, multi_close, multi_far)
- Oracle gap analysis
- Selector calibration (5 confidence bins)
- Diagnostics (collapse rate, complementarity, inter-hypothesis angle)
- Executed commands for reproducibility

### 3. Version Tracking (`pole_synth/version.py`)

```python
from pole_synth.version import __version__, get_version_info

print(__version__)  # "29I-rc1"
print(get_version_info())  # Full metadata dict
```

## Metrics Computed

| Metric | Description |
|--------|-------------|
| `set_err_mean/median` | Set-based pole error (considers both hypotheses) |
| `can_err_mean/median` | Canonical error (matches label convention) |
| `selector_accuracy` | How often selector picks the better hypothesis |
| `oracle_gap` | Selected@2 error minus Oracle@2 error |
| `collapse_rate` | Rate of hypothesis pairs within 5° |
| `complementarity_rate` | Rate where hyp1 beats hyp0 by ≥5° |

## Output JSON Schema

```json
{
  "version": "29I-rc1",
  "version_info": {...},
  "timestamp": "ISO-8601",
  "git_commit": "abc123...",
  "config": {...},
  "commands": ["..."],
  "n_folds": 3,
  "folds": {
    "fold_0": {...},
    "fold_1": {...},
    "fold_2": {...}
  },
  "aggregates": {
    "oracle_k2": {"overall": {...}, "by_label": {...}},
    "selected_learned_k2": {...},
    "selected_agreement_k2": {...},
    "naive0_k2": {...},
    "oracle_gap_learned": {"mean": X, "std": Y},
    "calibration": [...],
    "diagnostics": {...}
  }
}
```

## CLI Reference

```
usage: phase29i_paper_run.py [-h] --csv-dir DIR --artifacts-root DIR
                              [--selector-root DIR] [--folds N [N ...]]
                              [--device {cpu,cuda,mps}] [--seed N]
                              [--selector-mode {learned,agreement}]
                              [--label-mode {hybrid,canonical,set}]
                              [--train-selector] [--outdir DIR]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv-dir` | required | Directory with object CSV files |
| `--artifacts-root` | required | Root for fold artifacts |
| `--selector-root` | `{artifacts-root}/../selectors` | Selector checkpoints |
| `--folds` | `0 1 2` | Folds to evaluate |
| `--device` | `cuda` | Compute device |
| `--seed` | `42` | Random seed |
| `--selector-mode` | `learned` | Selector type |
| `--label-mode` | `hybrid` | Ground truth mode |
| `--train-selector` | False | Train selector if missing |
| `--outdir` | `results/phase29i_paper` | Output directory |

## Dependencies

No new dependencies added. Uses existing:
- numpy
- torch
- scipy
- scikit-learn

## Testing

Run Phase 29I tests:
```bash
pytest tests/test_phase29i_reporting.py -v
```

Run all tests:
```bash
pytest tests/ -v
```

## Backward Compatibility

- All Phase 29F/G/H functionality preserved
- Existing model checkpoints compatible
- Existing selector checkpoints compatible

## File Manifest

```
pole_synth/
├── version.py              # NEW: Version tracking
├── reporting.py            # NEW: Report builder
├── ambiguity_metrics.py    # Phase 29F metrics
├── ...

scripts/
├── phase29i_paper_run.py   # NEW: One-command paper run
├── phase29g_train_cv.py    # CV training (Phase 29G)
├── phase29h_train_selector.py  # Selector training (Phase 29H)
├── ...

tests/
├── test_phase29i_reporting.py  # NEW: Smoke tests
├── test_phase29h_*.py          # Phase 29H tests
├── ...
```

## Migration from Phase 29H

No migration required. Phase 29I adds new capabilities without modifying existing interfaces.

## Known Limitations

1. Paper run requires pre-trained models and optionally pre-trained selectors
2. Git commit hash only available when running from git repository
3. Calibration bins fixed at 5 (hardcoded)

## Next Steps

After verification:
1. Full paper run on production data
2. Generate final paper figures
3. Tag release: `v29I-rc1`
