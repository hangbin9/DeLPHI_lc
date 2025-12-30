#!/usr/bin/env python3
"""Run parity comparison between random and phase-stratified sampling."""

import subprocess
import json
from pathlib import Path

def run_experiment(seed: int, phase_stratified: int, outdir: str):
    """Run a single experiment."""
    cmd = [
        "python", "scripts/phase8_train_epoch_attention_grid.py",
        "--csv-dir", "DAMIT_csv_high",
        "--outdir", outdir,
        "--n-folds", "2",
        "--seeds", str(seed),
        "--epochs-max", "10",
        "--batch-size", "8",
        "--lr", "0.001",
        "--patience", "15",
        "--device", "cpu",
        "--n-epochs-per-sample", "5",
        "--n-tokens", "64",
        "--token-dim", "28",
        "--d-model", "64",
        "--n-layers", "2",
        "--n-poles", "4096",
        "--period-cache", "artifacts/period_cache_damit.json",
        "--allow-missing-periods", "0",
        "--phase-bins", "8",
        "--phase-stratified", str(phase_stratified),
        "--debug-phase-cov", "0",
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    # Read results
    results_path = Path(outdir) / "results.json"
    if results_path.exists():
        with open(results_path) as f:
            data = json.load(f)
        return data["mean_median"], data["std_median"]
    return None, None


def main():
    seeds = [42, 123, 456, 789, 1000, 2000, 3000, 4000, 5000, 6000]

    results_random = []
    results_strat = []

    for seed in seeds:
        print(f"=== Seed {seed} ===")

        # Random
        mean_r, std_r = run_experiment(
            seed, 0, f"artifacts/parity_seed{seed}_random"
        )
        if mean_r is not None:
            results_random.append(mean_r)
            print(f"  Random:     {mean_r:.2f}° ± {std_r:.2f}°")

        # Stratified
        mean_s, std_s = run_experiment(
            seed, 1, f"artifacts/parity_seed{seed}_strat"
        )
        if mean_s is not None:
            results_strat.append(mean_s)
            print(f"  Stratified: {mean_s:.2f}° ± {std_s:.2f}°")

        if mean_r is not None and mean_s is not None:
            delta = mean_s - mean_r
            print(f"  Delta:      {delta:+.2f}° (strat - random)")
        print()

    if results_random and results_strat:
        import numpy as np
        mean_r = np.mean(results_random)
        mean_s = np.mean(results_strat)
        std_r = np.std(results_random)
        std_s = np.std(results_strat)

        print("=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Random sampling:     {mean_r:.2f}° ± {std_r:.2f}° (n={len(results_random)})")
        print(f"Phase-stratified:    {mean_s:.2f}° ± {std_s:.2f}° (n={len(results_strat)})")
        print(f"Difference:          {mean_s - mean_r:+.2f}°")
        print()

        if mean_s < mean_r:
            print("=> Phase-stratified is BETTER")
        elif mean_s > mean_r:
            print("=> Random sampling is BETTER")
        else:
            print("=> No difference")


if __name__ == "__main__":
    main()
