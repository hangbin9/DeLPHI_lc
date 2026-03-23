# Data Format Guide

**lc_pipeline v1.0** - Unified Data Schema for Asteroid Lightcurve Analysis

---

## Quick Summary

**Do I need ground truth?** NO - only for training/evaluation
**Do I need to provide a period?** NO - it will be estimated automatically
**What's the minimum required?** Just lightcurve observations (times, brightnesses, geometry)

---

## Use Cases

### 1. Simple Inference (Most Common)

**You have**: Lightcurve observations
**You want**: Period + pole predictions
**You need**: Nothing else!

```json
{
  "object_id": "my_asteroid",
  "epochs": [
    {
      "epoch_id": 0,
      "observations": [
        {
          "time_jd": 2460000.5,
          "relative_brightness": 1.02,
          "sun_asteroid_vector": [0.5, -0.7, 0.5],
          "earth_asteroid_vector": [0.6, -0.6, 0.5]
        }
      ]
    }
  ]
}
```

**Result**: System estimates period automatically, then predicts 6-9 pole candidates (2-3 period aliases × 3 poles per alias). Half-period alias is only used if base period ≥ 8h.

---

### 2. Inference with Known Period

**You have**: Lightcurve + known rotation period
**You want**: Pole predictions (skip period estimation)
**Provide**: `period_hours` at top level

```json
{
  "object_id": "my_asteroid",
  "period_hours": 8.5,  // ← Known period (skip estimation)
  "epochs": [...]
}
```

**Result**: System uses your period (8.5h) and produces 9 pole candidates for aliases {8.5h, 17h, 4.25h}. For periods < 8h, only 6 candidates (base + double aliases).

**When to use**:
- You have a high-quality period from previous studies
- You want to save computation time
- You're testing period sensitivity

---

### 3. Training/Evaluation (Pole GT Only)

**You have**: Lightcurve + true pole solutions
**You want**: Train model or evaluate pole predictions
**Provide**: `ground_truth` with `pole_solutions`

```json
{
  "object_id": "asteroid_101",
  "epochs": [...],
  "ground_truth": {
    "pole_solutions": [
      {
        "lambda_deg": 210.5,
        "beta_deg": 62.3
      }
    ],
    "source": "DAMIT"
  }
}
```

**Result**: Period estimated automatically, pole predictions compared to ground truth

---

### 4. Evaluation (Period GT Only)

**You have**: Lightcurve + true rotation period
**You want**: Evaluate period estimation accuracy
**Provide**: `ground_truth` with `rotation_period_hours`

```json
{
  "object_id": "asteroid_101",
  "epochs": [...],
  "ground_truth": {
    "rotation_period_hours": 8.34,
    "source": "DAMIT"
  }
}
```

**Result**: Period estimation compared to ground truth, poles predicted normally

---

### 5. Training/Evaluation (Full GT)

**You have**: Lightcurve + true period + true poles
**You want**: Train model with full supervision
**Provide**: `ground_truth` with both period and poles

```json
{
  "object_id": "asteroid_101",
  "epochs": [...],
  "ground_truth": {
    "rotation_period_hours": 8.34,
    "pole_solutions": [
      {
        "lambda_deg": 210.5,
        "beta_deg": 62.3
      }
    ],
    "quality_flag": 3,
    "source": "DAMIT"
  }
}
```

**Result**: Both period and pole predictions compared to ground truth

---

## Field Reference

### Top-Level Fields

| Field | Required? | Type | Purpose |
|-------|-----------|------|---------|
| `object_id` | ✅ YES | string | Unique asteroid identifier |
| `epochs` | ✅ YES | list | Observation epochs |
| `period_hours` | ⬜ Optional | float | **Known period for inference** (skip estimation) |
| `ground_truth` | ⬜ Optional | object | **Ground truth for training/evaluation** |
| `format_version` | ⬜ Optional | string | Schema version (default: "1.0") |
| `coordinate_frame` | ⬜ Optional | string | Reference frame (default: "EQUATORIAL_J2000") |
| `metadata` | ⬜ Optional | dict | Additional asteroid metadata |

### Ground Truth Fields (All Optional)

**Important**: If providing `ground_truth`, you must provide **at least one** of period or poles.

| Field | Required? | Type | Purpose |
|-------|-----------|------|---------|
| `rotation_period_hours` | ⬜ Optional | float | True sidereal period (for evaluation) |
| `pole_solutions` | ⬜ Optional | list | True pole solutions (for training) |
| `quality_flag` | ⬜ Optional | int | DAMIT quality (1-5) |
| `source` | ⬜ Optional | string | Data source (e.g., "DAMIT", "LCDB") |
| `reference` | ⬜ Optional | string | Citation or DOI |

### Observation Fields

| Field | Required? | Type | Purpose |
|-------|-----------|------|---------|
| `time_jd` | ✅ YES | float | Julian Date |
| `relative_brightness` | ✅ YES | float | Normalized brightness (centered ~1.0) |
| `sun_asteroid_vector` | ✅ YES | [x,y,z] | Unit vector from asteroid to Sun |
| `earth_asteroid_vector` | ✅ YES | [x,y,z] | Unit vector from asteroid to Earth |
| `brightness_error` | ⬜ Optional | float | Measurement uncertainty (1-sigma) |
| `epoch_id` | ⬜ Optional | int | Epoch grouping identifier |

---

## Common Questions

### Q: What's the difference between `period_hours` and `ground_truth.rotation_period_hours`?

**A**:
- `period_hours` (top level) = **Known period for inference** - "I know the period, skip estimation"
- `ground_truth.rotation_period_hours` = **True period for evaluation** - "Compare your estimate to this"

**Example scenarios**:

```json
// Scenario 1: I know the period is 8.5h, predict the pole
{
  "period_hours": 8.5,  // ← Use this period
  "epochs": [...]
}

// Scenario 2: Estimate the period and compare to truth
{
  "epochs": [...],
  "ground_truth": {
    "rotation_period_hours": 8.5  // ← Compare estimate to this
  }
}

// Scenario 3: Use known period AND compare to truth (weird but allowed)
{
  "period_hours": 8.5,  // ← Use this for pole prediction
  "epochs": [...],
  "ground_truth": {
    "rotation_period_hours": 8.34  // ← Will report: your 8.5h is off by 0.16h
  }
}
```

### Q: Can I provide ground truth without any GT data?

**A**: NO - if you include `ground_truth`, you must provide **at least one** of:
- `rotation_period_hours` (for period evaluation)
- `pole_solutions` (for pole training/evaluation)

```json
// ❌ INVALID - ground_truth is empty
{
  "epochs": [...],
  "ground_truth": {}  // Error: must provide period or poles
}

// ✅ VALID - period GT only
{
  "epochs": [...],
  "ground_truth": {
    "rotation_period_hours": 8.5
  }
}

// ✅ VALID - pole GT only
{
  "epochs": [...],
  "ground_truth": {
    "pole_solutions": [{"lambda_deg": 210, "beta_deg": 62}]
  }
}
```

### Q: What if I have multiple pole solutions (ambiguity)?

**A**: That's common! DAMIT often has 2-4 solutions per asteroid. Provide them all:

```json
{
  "ground_truth": {
    "pole_solutions": [
      {"lambda_deg": 210.5, "beta_deg": 62.3},  // Solution 1
      {"lambda_deg": 185.2, "beta_deg": 55.1},  // Solution 2
      {"lambda_deg": 30.8, "beta_deg": -61.9}   // Solution 3 (antipode)
    ]
  }
}
```

The model will compute oracle error (minimum distance to any GT pole).

### Q: How do I provide pole coordinates?

**A**: You can use **either** ecliptic spherical OR Cartesian (or both):

```json
// Option 1: Ecliptic spherical (most common)
{"lambda_deg": 210.5, "beta_deg": 62.3}

// Option 2: Cartesian unit vector
{"cartesian": [0.35, -0.19, 0.92]}

// Option 3: Both (redundant but allowed)
{
  "lambda_deg": 210.5,
  "beta_deg": 62.3,
  "cartesian": [0.35, -0.19, 0.92]
}
```

**Coordinate systems**:
- **Ecliptic**: λ (longitude) = [0°, 360°), β (latitude) = [-90°, +90°]
- **Cartesian**: J2000 equatorial frame, unit vector (norm ≈ 1.0)

### Q: What's the "relative_brightness" field?

**A**: Normalized brightness values centered around 1.0.

**NOT magnitudes!** If you have magnitudes, convert:
```python
# Convert magnitude to relative brightness
flux = 10 ** (-mag / 2.5)
relative_brightness = flux / np.median(flux)  # Normalize to ~1.0
```

**What the model sees**:
- Values typically range 0.5 - 1.5 (relative to median)
- Centered at 1.0 for numerical stability
- Variations encode shape and orientation

### Q: What coordinate frame should I use?

**A**: **J2000 equatorial** (default). This is the standard for:
- Sun/Earth vectors from JPL Horizons
- Most asteroid databases

If you have ecliptic vectors, convert to J2000 equatorial before input.

---

## CSV Format (Alternative)

For users who prefer flat files:

```csv
time_jd,relative_brightness,sun_x,sun_y,sun_z,earth_x,earth_y,earth_z,epoch_id
2460000.5,1.02,0.5,-0.7,0.5,0.6,-0.6,0.5,0
2460000.6,0.98,0.5,-0.7,0.5,0.6,-0.6,0.5,0
2460001.5,1.05,0.5,-0.7,0.4,0.6,-0.6,0.4,1
```

**Required columns**:
- `time_jd`, `relative_brightness`
- `sun_x, sun_y, sun_z` (unit vector to Sun)
- `earth_x, earth_y, earth_z` (unit vector to Earth)

**Optional columns**:
- `brightness_error`, `epoch_id`
- `period_hours` (known period - use same value for all rows)

**Ground truth**: Provide in separate JSON file or header comment

See `lc_pipeline/schema.py` for full CSV schema documentation.

---

## Validation

The schema automatically validates:
- ✅ All required fields present
- ✅ Vectors are 3D and approximately unit norm (0.9 < norm < 1.1)
- ✅ Periods are physically reasonable (0.1h < P < 1000h)
- ✅ Angles in valid ranges
- ✅ At least one observation per epoch
- ✅ At least one epoch provided
- ✅ If ground_truth provided, at least one of period/poles included

**Validation errors**:
```python
from lc_pipeline.schema import LightcurveData

# This will raise validation error
data = LightcurveData(
    object_id="test",
    epochs=[],  # Error: must have at least one epoch
    ground_truth={}  # Error: must provide period or poles
)
```

---

## Complete Examples

### Example 1: Minimal Inference (No GT, Auto Period)

```json
{
  "object_id": "my_asteroid",
  "epochs": [
    {
      "epoch_id": 0,
      "observations": [
        {"time_jd": 2460000.5, "relative_brightness": 1.02,
         "sun_asteroid_vector": [0.554, -0.742, 0.479],
         "earth_asteroid_vector": [0.512, -0.578, 0.614]},
        {"time_jd": 2460000.6, "relative_brightness": 0.98,
         "sun_asteroid_vector": [0.554, -0.742, 0.479],
         "earth_asteroid_vector": [0.512, -0.578, 0.614]}
      ]
    }
  ]
}
```

### Example 2: Inference with Known Period

```json
{
  "object_id": "my_asteroid",
  "period_hours": 8.5,
  "epochs": [
    {
      "epoch_id": 0,
      "observations": [
        {"time_jd": 2460000.5, "relative_brightness": 1.02,
         "sun_asteroid_vector": [0.554, -0.742, 0.479],
         "earth_asteroid_vector": [0.512, -0.578, 0.614]}
      ]
    }
  ]
}
```

### Example 3: Training with Pole GT Only

```json
{
  "object_id": "asteroid_101",
  "epochs": [
    {
      "epoch_id": 0,
      "observations": [
        {"time_jd": 2433827.77, "relative_brightness": 0.988,
         "sun_asteroid_vector": [0.554, -0.742, 0.479],
         "earth_asteroid_vector": [0.512, -0.578, 0.614]}
      ]
    }
  ],
  "ground_truth": {
    "pole_solutions": [
      {"lambda_deg": 210.5, "beta_deg": 62.3},
      {"lambda_deg": 185.2, "beta_deg": 55.1}
    ],
    "quality_flag": 3,
    "source": "DAMIT"
  }
}
```

### Example 4: Full GT (Training)

```json
{
  "object_id": "asteroid_101",
  "epochs": [
    {
      "epoch_id": 0,
      "observations": [
        {"time_jd": 2433827.77, "relative_brightness": 0.988,
         "brightness_error": 0.02,
         "sun_asteroid_vector": [0.554, -0.742, 0.479],
         "earth_asteroid_vector": [0.512, -0.578, 0.614]}
      ]
    }
  ],
  "ground_truth": {
    "rotation_period_hours": 8.34,
    "pole_solutions": [
      {"lambda_deg": 210.5, "beta_deg": 62.3}
    ],
    "quality_flag": 3,
    "source": "DAMIT",
    "reference": "Durech et al. 2020"
  }
}
```

---

## See Also

- **[lc_pipeline/schema.py](../lc_pipeline/schema.py)** - Full schema definition with validation
- **[examples/](../examples/)** - Sample input files
- **[docs/USER_GUIDE.md](USER_GUIDE.md)** - Complete usage guide
- **[docs/API.md](API.md)** - API reference

---

**Last Updated**: 2026-01-12
**Schema Version**: 1.0
