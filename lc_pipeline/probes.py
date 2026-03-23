"""Conditioning probes to verify model is truly input-conditioned."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from lc_pipeline.training.losses_axisnet import antipode_angle

logger = logging.getLogger(__name__)


def inter_object_diversity_probe(
    model: torch.nn.Module,
    dataloader: object,
    n_samples: int = 10,
    device: str = 'cuda',
) -> Dict:
    """
    Check that different objects produce different pole sets.

    Returns:
        {
            'mean_inter_object_angle_deg': float,
            'min_inter_object_angle_deg': float,
            'pass': bool (True if mean > 2°),
        }
    """
    model.eval()
    angles = []

    with torch.no_grad():
        batch_count = 0
        for batch in dataloader:
            if batch_count >= n_samples:
                break

            tokens = batch['tokens'].to(device)
            mask = batch['mask'].to(device)

            poles, _ = model(tokens, mask)  # [B, 2, 3]
            B = poles.shape[0]

            # Compute inter-object angles
            for i in range(B):
                for j in range(i + 1, B):
                    # Angle between slot 0 of object i and slot 0 of object j
                    angle = antipode_angle(poles[i, 0:1], poles[j, 0:1]).item()
                    angles.append(angle)

            batch_count += 1

    if not angles:
        return {'mean_inter_object_angle_deg': 0.0, 'min_inter_object_angle_deg': 0.0, 'pass': False}

    mean_angle = float(np.mean(angles))
    min_angle = float(np.min(angles))

    return {
        'mean_inter_object_angle_deg': mean_angle,
        'min_inter_object_angle_deg': min_angle,
        'pass': mean_angle > 2.0,
    }


def input_sensitivity_probe(
    model: torch.nn.Module,
    dataloader: object,
    device: str = 'cuda',
) -> Dict:
    """
    Verify model output changes when input is modified.

    Tests:
    1. Shuffle magnitudes within window -> output should change
    2. Zero geometry features -> output should change

    Returns:
        {
            'magnitude_shuffle_changes_output': bool,
            'zero_geometry_changes_output': bool,
            'pass': bool (both should be True),
        }
    """
    model.eval()
    device_obj = torch.device(device)

    with torch.no_grad():
        for batch in dataloader:
            tokens_orig = batch['tokens'].to(device_obj)  # [B, W, T, 9]
            mask = batch['mask'].to(device_obj)

            poles_orig, _ = model(tokens_orig, mask)

            # Test 1: Shuffle magnitudes (feature 2)
            tokens_shuffled = tokens_orig.clone()
            for w in range(tokens_orig.shape[1]):
                for t in range(tokens_orig.shape[2]):
                    if mask[0, w, t] > 0.5:  # Valid token
                        tokens_shuffled[0, w, t, 2] += np.random.randn() * 0.5

            poles_shuffled, _ = model(tokens_shuffled, mask)
            mag_changes = float(torch.norm(poles_orig[0] - poles_shuffled[0])) > 0.01

            # Test 2: Zero geometry features (3-8)
            tokens_no_geom = tokens_orig.clone()
            tokens_no_geom[:, :, :, 3:9] = 0.0

            poles_no_geom, _ = model(tokens_no_geom, mask)
            geom_changes = float(torch.norm(poles_orig[0] - poles_no_geom[0])) > 0.01

            return {
                'magnitude_shuffle_changes_output': mag_changes,
                'zero_geometry_changes_output': geom_changes,
                'pass': mag_changes and geom_changes,
            }

    return {'magnitude_shuffle_changes_output': False, 'zero_geometry_changes_output': False, 'pass': False}


def run_conditioning_probes(
    model: torch.nn.Module,
    val_dataloader: object,
    outdir: Path,
    device: str = 'cuda',
) -> Dict:
    """Run all conditioning probes and save results."""
    probes_results = {
        'inter_object_diversity': inter_object_diversity_probe(model, val_dataloader, n_samples=10, device=device),
        'input_sensitivity': input_sensitivity_probe(model, val_dataloader, device=device),
    }

    # Overall pass/fail
    probes_results['overall_pass'] = (
        probes_results['inter_object_diversity'].get('pass', False) and
        probes_results['input_sensitivity'].get('pass', False)
    )

    # Save results
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    probe_file = outdir / 'conditioning_probes.json'
    with open(probe_file, 'w') as f:
        json.dump(probes_results, f, indent=2)

    logger.info(f"Conditioning probes: {probes_results['overall_pass']}")
    logger.info(f"  Inter-object diversity: {probes_results['inter_object_diversity'].get('mean_inter_object_angle_deg', 0):.2f}°")
    logger.info(f"  Input sensitivity: {probes_results['input_sensitivity'].get('pass', False)}")

    return probes_results
