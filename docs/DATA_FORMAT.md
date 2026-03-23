# Data Format Guide

## Quick Start

The pipeline accepts lightcurve data as **numpy arrays** with 8 columns. No special file format is required.

```python
import numpy as np
from lc_pipeline import analyze

# Each epoch is an (N, 8) numpy array
data = np.loadtxt("my_asteroid.csv", delimiter=',', skiprows=1)
epochs = [data]  # list of epoch arrays

result = analyze(epochs, "my_asteroid", fold=0)
```

---

## Input Format: 8-Column Array

Each observation is a row with 8 values:

| Column | Name | Description |
|--------|------|-------------|
| 0 | time | Julian Date |
| 1 | brightness | Relative brightness (centered around 1.0, NOT magnitudes) |
| 2 | sun_x | Sun-asteroid unit vector, x component |
| 3 | sun_y | Sun-asteroid unit vector, y component |
| 4 | sun_z | Sun-asteroid unit vector, z component |
| 5 | obs_x | Observer-asteroid unit vector, x component |
| 6 | obs_y | Observer-asteroid unit vector, y component |
| 7 | obs_z | Observer-asteroid unit vector, z component |

The coordinate frame is J2000 equatorial. Sun/observer vectors are currently not used by the model (geometry slots are set to zero), but must be present in the array for compatibility.

### CSV Example

```csv
time,mag,x,y,z,dx,dy,dz
2433827.771536,0.9882,-1.524,2.562,-1.655,-1.376,1.556,-1.655
2433827.809894,1.0045,-1.524,2.562,-1.655,-1.376,1.556,-1.655
2433827.848252,0.9931,-1.524,2.562,-1.655,-1.376,1.556,-1.655
```

### Multiple Epochs

Pass a list of arrays, one per observing epoch:

```python
epoch_1 = np.loadtxt("epoch1.csv", delimiter=',', skiprows=1)
epoch_2 = np.loadtxt("epoch2.csv", delimiter=',', skiprows=1)
epochs = [epoch_1, epoch_2]

result = analyze(epochs, "my_asteroid", fold=0)
```

If your data is in a single file, the pipeline will detect epochs automatically based on time gaps (>30 days).

---

## DAMIT Data

The pipeline was trained on data from the [DAMIT database](https://astro.troja.mff.cuni.cz/projects/damit/). DAMIT CSV files use exactly the 8-column format described above.

### Converting DAMIT Data

A converter is included for converting DAMIT data to the unified JSON schema used internally:

```python
from lc_pipeline.converters.damit_to_unified import convert_damit_to_unified

# Convert a single DAMIT CSV to unified format
unified_data = convert_damit_to_unified("data/damit_csv_qf_ge_3/asteroid_101.csv")
```

See `lc_pipeline/converters/damit_to_unified.py` for the full conversion logic. This serves as a reference if you need to adapt your own data format to work with the pipeline.

### Loading DAMIT CSVs Directly

For inference, you can load DAMIT CSVs directly as numpy arrays:

```python
import pandas as pd

df = pd.read_csv("asteroid_101.csv")
df = df.dropna()
data = df.values  # (N, 8) array
epochs = [data]
```

The `run_pole_prediction.py` script handles this automatically.

---

## Brightness Values

The pipeline expects **relative brightness** (flux-like values centered around 1.0), NOT astronomical magnitudes.

If you have magnitudes, convert them:

```python
import numpy as np

flux = 10 ** (-mag / 2.5)
relative_brightness = flux / np.median(flux)  # Normalize to ~1.0
```

---

## Unified JSON Schema (Advanced)

For programmatic use, the pipeline also supports a structured JSON format via the `lc_pipeline.schema` module. This is used internally and for data validation.

```python
from lc_pipeline.schema import LightcurveData, Epoch, Observation

data = LightcurveData(
    object_id="my_asteroid",
    epochs=[
        Epoch(observations=[
            Observation(
                time_jd=2460000.5,
                relative_brightness=1.02,
                sun_asteroid_vector=[0.554, -0.742, 0.479],
                earth_asteroid_vector=[0.512, -0.578, 0.614],
            ),
        ]),
    ],
)
```

See `lc_pipeline/schema.py` for the full schema definition with validation rules.

---

## See Also

- [examples/](../examples/) - Sample input files (CSV and JSON)
- [USER_GUIDE.md](USER_GUIDE.md) - Complete usage guide
- [API.md](API.md) - Python API reference
- `lc_pipeline/converters/damit_to_unified.py` - DAMIT format converter (reference implementation)
