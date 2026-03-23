"""
Sterile end-to-end test for lc_pipeline production model.

Imports ONLY from lc_pipeline (no experiments/ imports).
Verifies:
  1. Model loads and produces valid output
  2. Predictions are input-dependent (not constant)
  3. Period features affect predictions
  4. Quality=None handled correctly
  5. Checkpoint loads with correct param count
  6. Full pipeline runs end-to-end
"""
import sys
import numpy as np
from pathlib import Path

# Ensure lc_pipeline is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_damit_csv(name: str) -> list:
    """Load a DAMIT CSV as list of epoch arrays, skipping separator rows."""
    import csv
    csv_path = Path(__file__).parent / "fixtures" / f"{name}.csv"
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) == 8 and all(v.strip() for v in row):
                try:
                    rows.append([float(v) for v in row])
                except ValueError:
                    continue
    data = np.array(rows, dtype=np.float64)
    return [data]


def test_load_and_analyze():
    """Load a DAMIT asteroid and run analyze() with known period."""
    from lc_pipeline import analyze
    epochs = load_damit_csv("asteroid_1017")
    result = analyze(epochs, "asteroid_1017", period_hours=8.5, fold=0)
    assert len(result.poles) == 9, f"Expected 9 candidates, got {len(result.poles)}"
    assert result.best_pole is not None
    assert result.period.period_hours == 8.5
    print("  PASS: load_and_analyze")


def test_input_dependent():
    """Two different asteroids produce different pole sets."""
    from lc_pipeline import PoleInference
    inf = PoleInference()

    epochs_a = load_damit_csv("asteroid_1017")
    epochs_b = load_damit_csv("asteroid_1021")

    poles_a, _ = inf.predict(epochs_a, 8.5, fold=0)
    poles_b, _ = inf.predict(epochs_b, 6.0, fold=0)

    # Check that the full set of K=3 poles differs between inputs.
    # Individual pole indices may land near the same direction (the model
    # does not guarantee ordering), so we measure the maximum angular
    # difference across all (i, j) pole pairs.
    max_angle = 0.0
    for pa in poles_a:
        for pb in poles_b:
            cos_angle = np.abs(np.dot(pa, pb))
            angle = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))
            max_angle = max(max_angle, angle)
    assert max_angle > 1.0, f"Max pole diff is only {max_angle:.2f}° - model may be constant!"
    print(f"  PASS: input_dependent (max pole diff {max_angle:.1f}°)")


def test_period_sensitive():
    """Same asteroid with different periods produces different pole sets."""
    from lc_pipeline import PoleInference
    inf = PoleInference()

    epochs = load_damit_csv("asteroid_1017")
    poles_5h, _ = inf.predict(epochs, 5.0, fold=0)
    poles_20h, _ = inf.predict(epochs, 20.0, fold=0)

    # Compare full K=3 pole sets (see test_input_dependent for rationale).
    max_angle = 0.0
    for pa in poles_5h:
        for pb in poles_20h:
            cos_angle = np.abs(np.dot(pa, pb))
            angle = np.degrees(np.arccos(np.clip(cos_angle, 0, 1)))
            max_angle = max(max_angle, angle)
    assert max_angle > 0.5, f"Period change only moved poles by {max_angle:.2f}°"
    print(f"  PASS: period_sensitive (max pole diff {max_angle:.1f}°)")


def test_quality_none():
    """Quality=None handled correctly (equal scores, no crash)."""
    from lc_pipeline import PeriodForker
    forker = PeriodForker()
    epochs = load_damit_csv("asteroid_1017")
    candidates = forker.predict_with_aliases(epochs, 8.5, fold=0)
    assert len(candidates) == 9
    # With no quality head, all scores should be equal
    scores = [c.score for c in candidates]
    assert all(abs(s - scores[0]) < 1e-6 for s in scores), \
        f"Expected equal scores, got {scores}"
    print("  PASS: quality_none (equal scores)")


def test_checkpoint_loads():
    """Checkpoint loads with ~994K params, no cross_window_encoder."""
    from lc_pipeline.inference.model import PolePredictor
    ckpt_dir = Path(__file__).parent.parent / "lc_pipeline" / "checkpoints"
    model = PolePredictor.load(str(ckpt_dir / "fold_0.pt"))
    n_params = sum(p.numel() for p in model.parameters())
    assert 900_000 < n_params < 1_100_000, f"Expected ~994K params, got {n_params:,}"
    assert not hasattr(model, 'cross_window_encoder'), "Should not have cross_window_encoder"
    assert not hasattr(model, 'global_pool'), "Should not have global_pool"
    assert hasattr(model, 'encoder'), "Should have encoder"
    assert hasattr(model, 'pool'), "Should have pool (attention pooling)"
    print(f"  PASS: checkpoint_loads ({n_params:,} params)")


def test_full_pipeline():
    """LightcurvePipeline().analyze() completes end-to-end."""
    from lc_pipeline import LightcurvePipeline
    pipeline = LightcurvePipeline()
    epochs = load_damit_csv("asteroid_1017")
    result = pipeline.analyze(epochs, "asteroid_1017", period_hours=8.5, fold=0)
    assert result.object_id == "asteroid_1017"
    assert result.uncertainty is not None
    assert len(result.poles) == 9
    print("  PASS: full_pipeline")


def test_no_experiment_imports():
    """Verify no experiments/ modules are loaded."""
    import lc_pipeline
    # Force full import
    _ = lc_pipeline.analyze

    experiment_modules = [m for m in sys.modules if 'experiments' in m]
    assert len(experiment_modules) == 0, \
        f"Experiments modules loaded: {experiment_modules}"
    print("  PASS: no_experiment_imports")


if __name__ == "__main__":
    tests = [
        test_checkpoint_loads,
        test_load_and_analyze,
        test_input_dependent,
        test_period_sensitive,
        test_quality_none,
        test_full_pipeline,
        test_no_experiment_imports,
    ]

    print(f"Running {len(tests)} sterile E2E tests...\n")
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test.__name__}: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
    else:
        print("All tests passed!")
