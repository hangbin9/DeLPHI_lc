# CLAUDE.md - Project Instructions for Claude Code

## Project Overview

This is an asteroid lightcurve period prediction pipeline using physics-based methods (Lomb-Scargle + product-of-experts consensus). **No ML/DL for period estimation** - the pipeline is purely classical signal processing.

## Directory Structure

```
lc_pipeline/           # Core library
├── config.py          # PeriodConfig, ColumnConfig, PhysicalAliasConfig
├── data.py            # LightcurveEpoch, AsteroidLightcurves, loaders
├── period_search.py   # Lomb-Scargle implementation
├── posterior.py       # Softmax, clustering, credible intervals
├── consensus.py       # ConsensusEngine (main entry point)
├── alias_ellipsoid_resolver.py  # EXPERIMENTAL, disabled by default
├── metrics.py         # Alias-aware evaluation metrics
├── io_utils.py        # File I/O helpers
└── plotting.py        # Diagnostic visualizations

scripts/               # CLI tools
├── run_period_inference.py
├── evaluate_period_predictions.py
├── compare_resolver_vs_baseline.py
└── ...

tests/                 # Unit tests (pytest)
├── test_ls_and_consensus.py
├── test_physical_alias_resolver_synthetic.py
└── ...

data/                  # Generated data files
├── damit_manifest.csv
└── damit_groundtruth.csv

DAMIT_csv_high/        # Raw DAMIT lightcurve CSVs (not in git)
```

## Key Conventions

### Data Format (DAMIT)
- Time column: `jd` (Julian Date)
- Flux column: `relative_brightness` → converted to magnitude via `mag = -2.5 * log10(flux)`
- Ground truth period: `rot_per` column (already in hours)
- Default magnitude error: 0.02 mag (when not provided)

### Units
- **Periods**: Always in hours internally
- **Time**: Days (JD or MJD)
- **Frequencies**: cycles/day for Lomb-Scargle

### Alias Awareness
- Metrics consider {P, 0.5P, 2P} alias family
- `alias_aware_relative_error()` takes minimum error over alias family
- Factor-of-2 aliases are expected due to double-peaked asteroid lightcurves

## Important Design Decisions

1. **PhysicalAliasConfig.enabled = False by default**
   - The ellipsoid resolver is EXPERIMENTAL
   - Synthetic tests show it fails for symmetric lightcurves
   - Only enable if you understand the limitations

2. **No train/test leakage**
   - All epochs from same object must be in same split
   - Evaluation only compares predictions to ground truth
   - No fitting or tuning based on ground truth

3. **Purely descriptive uncertainty**
   - Returns raw posterior + credible intervals
   - No "green/gray" classification baked in
   - Downstream code decides interpretation

## Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_ls_and_consensus.py -v

# Synthetic resolver validation (expected to FAIL - this is correct)
pytest tests/test_physical_alias_resolver_synthetic.py -v
```

## Common Commands

```bash
# Build manifest from DAMIT CSVs
python -m scripts.build_manifest_from_dir --root-dir DAMIT_csv_high/ --out-manifest data/damit_manifest.csv

# Extract ground truth
python -m scripts.extract_groundtruth_from_damit --manifest data/damit_manifest.csv --out-groundtruth data/damit_groundtruth.csv

# Run period inference
python -m scripts.run_period_inference --manifest data/damit_manifest.csv --out-predictions results/predictions.csv

# Evaluate predictions
python -m scripts.evaluate_period_predictions --predictions results/predictions.csv --groundtruth data/damit_groundtruth.csv

# Compare resolver vs baseline
python -m scripts.compare_resolver_vs_baseline --manifest data/damit_manifest.csv --groundtruth data/damit_groundtruth.csv
```

## Performance Baseline

On 170 DAMIT objects:
- Alias-aware Acc@5%: 74.1%
- Alias-aware Acc@10%: 77.1%
- Alias-aware Acc@20%: 84.1%

## Do NOT

- Do not add ML/DL to the period prediction pipeline
- Do not modify existing LS or posterior math without good reason
- Do not enable the physical alias resolver by default
- Do not commit `__pycache__/`, `*.pyc`, or large data files
