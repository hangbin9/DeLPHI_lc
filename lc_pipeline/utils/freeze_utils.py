#!/usr/bin/env python3
"""
Utilities for Progressive Unfreezing During Transfer Learning.

Implements layer-wise freezing/unfreezing schedules to prevent catastrophic
forgetting while fine-tuning pretrained models on real data.

Strategy:
- Start with most layers frozen (prevent weight destruction)
- Progressively unfreeze layers from output to input
- Gradually increase learning rates
"""

import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class LayerGroupManager:
    """
    Organizes model layers into semantic groups for progressive unfreezing.

    For GeoHierK3Transformer, groups might be:
    - embedding_layers: Input embeddings
    - window_encoder_layers: Core feature extraction
    - cross_encoder_layers: Cross-epoch aggregation
    - head_layers: Task-specific heads
    """

    def __init__(self, model: nn.Module):
        """
        Initialize layer group manager.

        Args:
            model: PyTorch model to manage
        """
        self.model = model
        self.groups = self._create_layer_groups()

    def _create_layer_groups(self) -> Dict[str, List[str]]:
        """
        Organize model parameters into semantic groups.

        Returns:
            Dict mapping group name to list of parameter names
        """
        groups = {
            'embedding': [],
            'backbone': [],  # Main feature extraction
            'aggregation': [],  # Cross-epoch aggregation
            'heads': [],  # Prediction heads
        }

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Embedding,)):
                # Embeddings
                groups['embedding'].extend(self._get_param_names(name))
            elif 'encoder' in name.lower() or 'backbone' in name.lower():
                # Feature extraction layers
                groups['backbone'].extend(self._get_param_names(name))
            elif 'cross' in name.lower() or 'aggregate' in name.lower():
                # Aggregation layers
                groups['aggregation'].extend(self._get_param_names(name))
            elif any(head in name.lower() for head in ['head', 'classifier', 'predictor']):
                # Task-specific heads
                groups['heads'].extend(self._get_param_names(name))
            else:
                # Default to backbone if unclassified
                groups['backbone'].extend(self._get_param_names(name))

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _get_param_names(self, module_name: str) -> List[str]:
        """Get all parameter names under a module."""
        param_names = []
        for name, param in self.model.named_parameters():
            if name.startswith(module_name):
                param_names.append(name)
        return param_names

    def get_groups(self) -> Dict[str, List[str]]:
        """Get layer groups."""
        return self.groups

    def print_groups(self):
        """Print summary of layer groups."""
        print("Layer Groups:")
        print("=" * 80)
        total_params = sum(p.numel() for p in self.model.parameters())

        for group_name, param_names in self.groups.items():
            if not param_names:
                continue
            n_params = sum(
                self.model.state_dict()[name].numel()
                for name in param_names
                if name in self.model.state_dict()
            )
            pct = 100 * n_params / total_params
            print(f"{group_name:15} {len(param_names):4} params ({pct:5.1f}%)")

        print("=" * 80)


def freeze_parameters(model: nn.Module, param_names: List[str]):
    """
    Freeze specific parameters by name.

    Args:
        model: PyTorch model
        param_names: List of parameter names to freeze
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if name in param_names:
            param.requires_grad = False
            frozen_count += 1

    logger.info(f"Froze {frozen_count} parameters")


def unfreeze_parameters(model: nn.Module, param_names: List[str]):
    """
    Unfreeze specific parameters by name.

    Args:
        model: PyTorch model
        param_names: List of parameter names to unfreeze
    """
    unfrozen_count = 0
    for name, param in model.named_parameters():
        if name in param_names:
            param.requires_grad = True
            unfrozen_count += 1

    logger.info(f"Unfroze {unfrozen_count} parameters")


def freeze_all_except(model: nn.Module, unfreeze_names: List[str]):
    """
    Freeze all parameters except those in the list.

    Args:
        model: PyTorch model
        unfreeze_names: List of parameter names to keep unfrozen
    """
    unfreeze_set = set(unfreeze_names)
    frozen_count = 0

    for name, param in model.named_parameters():
        if name not in unfreeze_set:
            param.requires_grad = False
            frozen_count += 1
        else:
            param.requires_grad = True

    logger.info(f"Froze {frozen_count} parameters, keeping {len(unfreeze_set)} unfrozen")


def get_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Get counts of trainable and frozen parameters.

    Returns:
        (n_trainable, n_frozen)
    """
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return n_trainable, n_frozen


def print_trainable_status(model: nn.Module):
    """Print human-readable summary of trainable/frozen parameters."""
    n_trainable, n_frozen = get_trainable_parameters(model)
    total = n_trainable + n_frozen

    print("=" * 80)
    print("Model Trainable Status")
    print("=" * 80)
    print(f"Trainable: {n_trainable:,} ({100*n_trainable/total:.1f}%)")
    print(f"Frozen:    {n_frozen:,} ({100*n_frozen/total:.1f}%)")
    print(f"Total:     {total:,}")
    print("=" * 80)


class ProgressiveUnfreezeSchedule:
    """
    Defines a schedule for progressive unfreezing during fine-tuning.

    Example:
    ```python
    schedule = ProgressiveUnfreezeSchedule(
        total_epochs=150,
        strategy='bottom_up',
    )

    for epoch in range(150):
        frozen_groups = schedule.get_frozen_groups(epoch)
        freeze_groups(model, frozen_groups)
    ```
    """

    def __init__(
        self,
        total_epochs: int = 150,
        strategy: str = "bottom_up",
        initial_frozen_groups: List[str] = None,
    ):
        """
        Initialize unfreezing schedule.

        Args:
            total_epochs: Total number of training epochs
            strategy: 'bottom_up' (unfreeze from output to input),
                     'top_down' (unfreeze from input to output),
                     'exponential' (exponential unfreezing)
            initial_frozen_groups: Groups to keep frozen initially
        """
        self.total_epochs = total_epochs
        self.strategy = strategy
        self.initial_frozen_groups = initial_frozen_groups or []

    def get_frozen_groups(self, epoch: int, all_groups: List[str]) -> List[str]:
        """
        Get list of groups that should be frozen at given epoch.

        Args:
            epoch: Current epoch (0-indexed)
            all_groups: List of all available groups

        Returns:
            List of group names to keep frozen
        """
        if self.strategy == "bottom_up":
            return self._get_bottom_up_frozen(epoch, all_groups)
        elif self.strategy == "top_down":
            return self._get_top_down_frozen(epoch, all_groups)
        elif self.strategy == "exponential":
            return self._get_exponential_frozen(epoch, all_groups)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _get_bottom_up_frozen(self, epoch: int, all_groups: List[str]) -> List[str]:
        """Unfreeze from output layers to input (heads first)."""
        # Order: heads → aggregation → backbone → embedding
        unfreeze_order = ['heads', 'aggregation', 'backbone', 'embedding']

        n_to_unfreeze = (epoch // (self.total_epochs // len(unfreeze_order))) + 1
        n_to_unfreeze = min(n_to_unfreeze, len(unfreeze_order))

        groups_to_unfreeze = set(unfreeze_order[:n_to_unfreeze])
        return [g for g in all_groups if g not in groups_to_unfreeze]

    def _get_top_down_frozen(self, epoch: int, all_groups: List[str]) -> List[str]:
        """Unfreeze from input layers to output (embeddings first)."""
        # Order: embedding → backbone → aggregation → heads
        unfreeze_order = ['embedding', 'backbone', 'aggregation', 'heads']

        n_to_unfreeze = (epoch // (self.total_epochs // len(unfreeze_order))) + 1
        n_to_unfreeze = min(n_to_unfreeze, len(unfreeze_order))

        groups_to_unfreeze = set(unfreeze_order[:n_to_unfreeze])
        return [g for g in all_groups if g not in groups_to_unfreeze]

    def _get_exponential_frozen(self, epoch: int, all_groups: List[str]) -> List[str]:
        """Exponentially unfreeze layers."""
        # Fraction of epochs complete
        progress = epoch / self.total_epochs
        # Exponential unfreeze (accelerating)
        unfreeze_fraction = min(1.0, 2 * progress ** 0.5)

        n_groups = len(all_groups)
        n_to_unfreeze = int(n_groups * unfreeze_fraction)
        n_to_unfreeze = max(0, min(n_to_unfreeze, n_groups))

        # Unfreeze from output to input (bottom-up)
        unfreeze_order = ['heads', 'aggregation', 'backbone', 'embedding']
        groups_to_unfreeze = set(unfreeze_order[:n_to_unfreeze])

        return [g for g in all_groups if g not in groups_to_unfreeze]


def apply_unfreezing_schedule(
    model: nn.Module,
    layer_groups: Dict[str, List[str]],
    epoch: int,
    schedule: ProgressiveUnfreezeSchedule,
):
    """
    Apply progressive unfreezing at given epoch.

    Args:
        model: PyTorch model
        layer_groups: Dict of group_name → parameter_names
        epoch: Current epoch
        schedule: ProgressiveUnfreezeSchedule instance
    """
    frozen_groups = schedule.get_frozen_groups(
        epoch, list(layer_groups.keys())
    )

    # Freeze specified groups
    for group_name, param_names in layer_groups.items():
        if group_name in frozen_groups:
            freeze_parameters(model, param_names)
        else:
            unfreeze_parameters(model, param_names)


if __name__ == "__main__":
    # Demo: show unfreezing schedule
    print("Progressive Unfreezing Schedules")
    print("=" * 80)

    schedule = ProgressiveUnfreezeSchedule(
        total_epochs=150,
        strategy="bottom_up",
    )

    all_groups = ["embedding", "backbone", "aggregation", "heads"]

    print(f"{'Epoch':<8} {'Frozen Groups':<60}")
    print("-" * 80)

    for epoch in [0, 30, 60, 90, 120, 150]:
        frozen = schedule.get_frozen_groups(epoch, all_groups)
        print(f"{epoch:<8} {', '.join(frozen):<60}")
