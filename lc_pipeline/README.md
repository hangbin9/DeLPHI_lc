# lc_pipeline Package Notes

This directory contains the Python package for the DeLPHI asteroid lightcurve pipeline.

For most users, start with the repository-level docs instead of this file:

- [../README.md](../README.md)
- [../docs/USER_GUIDE.md](../docs/USER_GUIDE.md)

## What The Package Provides

- end-to-end `analyze()` API
- pre-trained fold checkpoints under `lc_pipeline/checkpoints/`
- classical period estimation
- multi-candidate pole prediction
- unified JSON schema and loaders

## Quick Python Example

```python
import pandas as pd
from lc_pipeline import analyze

df = pd.read_csv("examples/asteroid_101.csv", comment="#")
result = analyze([df.values], "asteroid_101", fold=0)

print(result.period.period_hours)
print(result.best_pole.lambda_deg, result.best_pole.beta_deg)
```

## Related Docs

- [../docs/API.md](../docs/API.md)
- [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md)
- [../docs/DATA_FORMAT.md](../docs/DATA_FORMAT.md)
