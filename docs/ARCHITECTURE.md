# DeLPHI Architecture

Technical overview of the `lc_pipeline` package and how it relates to the published DeLPHI manuscript.

## High-Level Design

The production pipeline has two stages:

1. **Period estimation** with a classical Lomb-Scargle based method
2. **Pole prediction** with a Transformer model that returns multiple candidate poles

This is a hybrid classical + ML system by design:

- the period search is handled by a transparent classical method
- the pole search is handled by a learned model that narrows the candidate sky region

## Published Evaluation Context

The manuscript evaluates the production approach on:

- **174 DAMIT asteroids** in 5-fold cross-validation
- **163 ZTF asteroids** for external validation

Published headline results:

- CV mean oracle error: **19.02° ± 2.68°**
- CV pooled median oracle error: **16.61°**
- End-to-end mean oracle error with estimated periods: **18.90°**
- ZTF mean oracle error: **18.82° ± 1.02°**

## Stage 1: Period Estimation

The period stage lives under `lc_pipeline.period`.

Core behavior:

- per-epoch Lomb-Scargle periodograms
- multi-epoch posterior fusion
- alias-aware handling of factor-of-two ambiguities
- uncertainty summary from the fused posterior

Search range:

- production search range: **2 to 200 hours**

Published manuscript result:

- median alias-aware relative error: **5.3%**

## Stage 2: Pole Prediction

The pole stage lives under `lc_pipeline.inference` and `lc_pipeline.models`.

### Input representation

Each token is a **13-dimensional** feature vector:

- temporal features: `t_norm`, `dt_norm`
- brightness feature: robust MAD-normalized brightness
- geometry slots: 6 reserved dimensions, **set to zero in the production model**
- period features: 4 dimensions derived from period and rotation phase

At inference:

- the lightcurve is split into up to **8 windows**
- a new window starts when the time gap exceeds **1 day**
- each window is capped at **256 tokens**

### Model

The production pole model is the single-encoder Transformer implemented in `lc_pipeline/models/geo_hier_k3_transformer.py`.

Key properties:

- about **1.0 million parameters**
- embedding dimension: **128**
- attention heads: **4**
- encoder layers: **4**
- attention pooling within each window
- mean pooling across windows
- **K = 3** pole output heads

The model outputs **3 candidate poles**, not one guaranteed final solution.

## Alias Expansion At Inference

After period estimation, the package tests pole prediction at:

- `P`
- `2P`
- `P/2` when the base period is at least `8 h`

That produces **6 to 9** total candidates.

Important interpretation detail:

- this is why the package returns multiple pole candidates even when the base model has only 3 output heads

## Candidate Ranking And Scores

The production API returns a `score` for each candidate and sorts candidates by that value.

Use that cautiously:

- the manuscript evaluates the model with an **oracle** metric, not a guaranteed learned selector
- candidate scores are helpful for inspection
- users should still review multiple candidates and antipodes

## Geometry Inputs

The input schema includes Sun and observer vectors, but the production production tokenization sets the six geometry feature slots to zero.

This matches the current manuscript and code:

- the slots are reserved for future work
- the production model currently relies on the brightness/time/period pathway rather than explicit geometry features

## Training Setup In The Manuscript

The manuscript describes:

- 5-fold cross-validation split by asteroid ID
- 138 to 141 training asteroids per fold
- 35 to 36 validation asteroids per fold
- multi-epoch expansion to **2,987** training samples from the 174-asteroid set

This is the production research setup behind the published results.

## Known Limits

- The package narrows the search space; it does not replace full physical inversion.
- The main published accuracy is an oracle metric over multiple candidates.
- Equator-on asteroids are substantially harder than pole-on asteroids.
- Best performance depends on multi-apparition coverage, not only on dense sampling within one night.

## Related Docs

- [docs/USER_GUIDE.md](USER_GUIDE.md): beginner workflow
- [docs/API.md](API.md): public API
- [docs/DATA_FORMAT.md](DATA_FORMAT.md): supported input formats
- [paper/manuscript.tex](../paper/manuscript.tex): published method description and results
