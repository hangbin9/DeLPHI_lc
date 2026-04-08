# Example Files

Example inputs for the `lc_pipeline` inference workflow.

## Files

| File | Purpose |
|------|---------|
| `asteroid_101.csv` | simplest CSV example for the CLI and Python API |
| `asteroid_101.json` | unified JSON example with ground-truth metadata |
| `inference_only.json` | unified JSON example without ground truth |

## Fastest Way To Test The Package

```bash
python run_pole_prediction.py --input examples/asteroid_101.csv
```

## Python Example

```python
import pandas as pd
from lc_pipeline import analyze

df = pd.read_csv("examples/asteroid_101.csv", comment="#")
result = analyze([df.values], "asteroid_101", fold=0)

print(f"Period: {result.period.period_hours:.2f} h")
print(f"Top candidate: λ={result.best_pole.lambda_deg:.1f}°, β={result.best_pole.beta_deg:.1f}°")
print(f"Returned candidates: {len(result.poles)}")
```

## Notes

- The example files are for **inference** and format validation.
- Ground truth in the JSON example is there for inspection, not because inference requires it.

## Related Docs

- [../README.md](../README.md)
- [../docs/USER_GUIDE.md](../docs/USER_GUIDE.md)
- [../docs/DATA_FORMAT.md](../docs/DATA_FORMAT.md)
