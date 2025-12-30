#!/usr/bin/env python3
"""
Pre-flight check for fold0_hierarchical_vs_flat.py comparison.

Verifies:
1. Dataset files exist and are valid
2. Model imports work
3. CUDA/CPU device is available
4. Architectures can forward pass
"""

import sys
import pickle
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, '/mnt/d/Downloads/Colab Notebooks')

print("="*70)
print("PRE-FLIGHT CHECK: Fold 0 Fair Comparison")
print("="*70)
print()

checks_passed = 0
checks_total = 0

# Check 1: Dataset files exist
print("Check 1: Dataset files")
print("-" * 70)
checks_total += 1

dataset_dir = Path('/tmp/phase1_pole_dataset/fold_0')
train_file = dataset_dir / 'train_asteroids.pkl'
test_file = dataset_dir / 'test_asteroids.pkl'

if train_file.exists() and test_file.exists():
    print(f"✅ Found dataset files:")
    print(f"   {train_file}")
    print(f"   {test_file}")

    # Load and verify
    try:
        with open(train_file, 'rb') as f:
            train_asteroids = pickle.load(f)
        with open(test_file, 'rb') as f:
            test_asteroids = pickle.load(f)

        print(f"✅ Loaded successfully:")
        print(f"   Train: {len(train_asteroids)} asteroids")
        print(f"   Test: {len(test_asteroids)} asteroids")

        # Check data structure
        test_ast = test_asteroids[0]
        required_keys = ['apparition_features', 'pole_true', 'asteroid_id']
        if all(k in test_ast for k in required_keys):
            print(f"✅ Data structure valid (has {required_keys})")
            app_feat = test_ast['apparition_features']
            if isinstance(app_feat, list) and len(app_feat) > 0:
                feat_shape = app_feat[0].shape
                if feat_shape[1] == 14:
                    print(f"✅ Features are 14D: {feat_shape}")
                    checks_passed += 1
                else:
                    print(f"❌ Features are {feat_shape[1]}D, expected 14D")
            else:
                print(f"❌ apparition_features not a list or empty")
        else:
            print(f"❌ Missing required keys: {required_keys}")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
else:
    print(f"❌ Dataset files not found:")
    print(f"   {train_file}: {'✓' if train_file.exists() else '✗'}")
    print(f"   {test_file}: {'✓' if test_file.exists() else '✗'}")

print()

# Check 2: Model imports
print("Check 2: Model imports")
print("-" * 70)
checks_total += 1

try:
    from scripts.train_phase1_probabilistic_pole import (
        AsteroidApparitionDataset,
        collate_apparitions,
        TransformerEncoderWithVMF,
    )
    print("✅ Imported TransformerEncoderWithVMF (flat model)")

    from scripts.train_phase1_hierarchical import (
        HierarchicalTransformerWithVMF,
        collate_apparitions_hierarchical
    )
    print("✅ Imported HierarchicalTransformerWithVMF (hierarchical model)")

    from scripts.losses import AntipodeAwareNLL
    print("✅ Imported AntipodeAwareNLL loss")

    checks_passed += 1
except ImportError as e:
    print(f"❌ Import failed: {e}")

print()

# Check 3: Device availability
print("Check 3: Device availability")
print("-" * 70)
checks_total += 1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ Using device: {device}")

if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"   (CPU mode - expect ~2-3 hours for 9 training runs)")

checks_passed += 1

print()

# Check 4: Architecture forward pass
print("Check 4: Architecture forward pass")
print("-" * 70)
checks_total += 1

try:
    # Dummy data
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Test flat model
    print("Testing Flat-512...")
    flat_model = TransformerEncoderWithVMF(
        d_model=128, n_heads=4, n_layers=2, n_components=3, input_dim=14
    ).to(device)

    x_flat = torch.randn(batch_size, 128, 14).to(device)
    mask_flat = torch.ones(batch_size, 128).to(device)

    with torch.no_grad():
        out_flat = flat_model(x_flat, mask_flat)

    if 'mu' in out_flat and out_flat['mu'].shape == (batch_size, 3, 3):
        print(f"✅ Flat model forward pass OK: mu={out_flat['mu'].shape}, weight={out_flat['weight'].shape}")
    else:
        print(f"❌ Flat model output shape mismatch")

    # Test hierarchical model
    print("Testing Hier-1024...")
    hier_model = HierarchicalTransformerWithVMF(
        d_model=128, n_heads=4, n_layers=2, n_components=3, input_dim=14
    ).to(device)

    x_hier = torch.randn(batch_size, 8, 128, 14).to(device)
    mask_tok = torch.ones(batch_size, 8, 128).to(device)
    mask_app = torch.ones(batch_size, 8).to(device)

    with torch.no_grad():
        out_hier = hier_model(x_hier, mask_tok, mask_app)

    if 'mu' in out_hier and out_hier['mu'].shape == (batch_size, 3, 3):
        print(f"✅ Hier model forward pass OK: mu={out_hier['mu'].shape}, weight={out_hier['weight'].shape}")
    else:
        print(f"❌ Hier model output shape mismatch")

    checks_passed += 1

except Exception as e:
    print(f"❌ Architecture test failed: {e}")
    import traceback
    traceback.print_exc()

print()

# Summary
print("="*70)
print(f"PRE-FLIGHT CHECK: {checks_passed}/{checks_total} passed")
print("="*70)
print()

if checks_passed == checks_total:
    print("✅ ALL CHECKS PASSED - Ready to run comparison!")
    print()
    print("Run with:")
    print("  cd /mnt/d/Downloads/Colab\\ Notebooks")
    print("  python scripts/fold0_hierarchical_vs_flat.py")
    print()
    print("Expected runtime: ~3-4 hours on GPU, ~6-9 hours on CPU")
    sys.exit(0)
else:
    print(f"❌ {checks_total - checks_passed} checks failed - Fix issues before running")
    sys.exit(1)
