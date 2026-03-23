# lc_pipeline: Asteroid Lightcurve Analysis

Asteroid period and pole estimation from multi-epoch lightcurves.

## Quick Start

```python
from lc_pipeline import analyze

result = analyze(epochs, "asteroid_1017", fold=0)

print(f"Period: {result.period.period_hours:.2f} h")
print(f"Best pole: lambda={result.best_pole.lambda_deg:.1f}, beta={result.best_pole.beta_deg:.1f}")

# All candidates (6-9 total, unranked)
for pole in result.poles:
    print(f"  lambda={pole.lambda_deg:.1f}, beta={pole.beta_deg:.1f}, alias={pole.alias}")
```

## Performance

**5-Fold Cross-Validation on 174 DAMIT QF>=3 asteroids:**

| Metric | Value |
|--------|-------|
| Mean Oracle Error | **19.02 deg** |
| Pooled Median | **16.61 deg** |
| Across-Fold Std | +/- 2.68 deg |
| ZTF External (163 asteroids) | **18.82 deg** |

## Architecture

- **Stage 1**: Period estimation (Lomb-Scargle + Bayesian consensus)
- **Stage 2**: Pole prediction (Transformer, d_model=128, 4 layers, 4 heads, ~994K params)
- 13-feature tokenization (3 temporal + 6 geometry zeros + 4 period)
- K=3 unranked pole candidates per period alias
- Period alias expansion (P, 2P, P/2)

## Checkpoints

| File | Oracle Mean |
|------|------------|
| fold_0.pt | 19.51 deg |
| fold_1.pt | 14.88 deg |
| fold_2.pt | 18.32 deg |
| fold_3.pt | 22.05 deg |
| fold_4.pt | 20.34 deg |

See the top-level README.md for full documentation.
