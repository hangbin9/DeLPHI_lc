"""Loss functions for asteroid pole prediction.

Includes:
- Core utilities: vectorize_solutions, antipode_angle
- Oracle loss: oracle_k3_loss (softmin over K=3 hypotheses)
- Quality loss: gap_weighted_quality_loss_k3 (optional selector head)
- Anti-collapse: batch_variance_loss, similarity_matching_loss, continuous_diversity_loss
- Combined loss: combined_loss (production 4-term loss)
"""

import logging
from typing import List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# UTILITIES: Convert solutions_list to vectorized format
# ============================================================================

def vectorize_solutions(
    solutions_list: List[Optional[torch.Tensor]],
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert solutions_list (Python list of variable-length tensors) to vectorized format.

    Args:
        solutions_list: List of [S, 3] solution tensors, where S can vary per batch item.
                       Items can be None or empty.
        device: Target device. If None, infers from first non-None solution or defaults to CPU.

    Returns:
        solutions_padded: [B, max_sols, 3] padded tensor
        solutions_mask: [B, max_sols] boolean mask indicating valid solutions
    """
    B = len(solutions_list)

    # Infer device if not provided
    if device is None:
        device = torch.device('cpu')
        for sol in solutions_list:
            if sol is not None:
                device = sol.device
                break

    # Find max number of solutions
    max_sols = 0
    for sols in solutions_list:
        if sols is not None:
            max_sols = max(max_sols, sols.shape[0])

    if max_sols == 0:
        # No valid solutions - return empty tensors
        return torch.zeros(B, 1, 3, dtype=torch.float32, device=device), \
               torch.zeros(B, 1, dtype=torch.bool, device=device)

    # Create padded tensor and mask
    solutions_padded = torch.zeros(B, max_sols, 3, dtype=torch.float32, device=device)
    solutions_mask = torch.zeros(B, max_sols, dtype=torch.bool, device=device)

    for b, sols in enumerate(solutions_list):
        if sols is not None and len(sols) > 0:
            n = sols.shape[0]
            solutions_padded[b, :n] = sols
            solutions_mask[b, :n] = True

    return solutions_padded, solutions_mask


# ============================================================================
# CORE: Antipode-aware angle computation
# ============================================================================

def antipode_angle(p: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Antipode-aware angle between prediction and solution (TRAINING version).

    Uses clamp(-0.99, 0.99) for gradient stability during backpropagation.
    This creates an 8.11° floor (arccos(0.99)) which is acceptable for loss
    computation but NOT for evaluation metrics.  Use eval_antipode_angle()
    for metric computation.

    Args:
        p: [*, 3] prediction (assumed unit vectors)
        s: [*, 3] solution (assumed unit vectors)

    Returns:
        [*] angles in degrees, accounting for antipodal symmetry
    """
    # Compute cosine of angle
    cos_angle = torch.sum(p * s, dim=-1)  # [*]

    # Clamp to avoid numerical issues with arccos AND to ensure stability during backprop
    # Wider range helps avoid gradient instability from acos at boundaries
    # Range: [-0.99, 0.99] provides stable gradient region
    cos_angle_clamped = torch.clamp(cos_angle, -0.99, 0.99)  # [*]

    # For antipode awareness: use absolute value of cosine (squared to avoid abs() gradient issues)
    # angle(p,s) == angle(p,-s) because both have same |cos(angle)|
    # Use cos^2 which is smooth: d/dx(x^2) = 2x (no discontinuity at zero)
    cos_angle_squared = cos_angle_clamped * cos_angle_clamped  # [*]

    # Compute angle from cos^2
    # acos(sqrt(x)) is stable when x in (0, 1)
    # Add epsilon to avoid sqrt(0)
    eps = 1e-6
    cos_angle_squared_safe = torch.clamp(cos_angle_squared, eps, 1.0)
    abs_cos_angle = torch.sqrt(cos_angle_squared_safe)  # Equivalent to |cos_angle|

    # Compute angle from absolute cosine
    angle_rad = torch.acos(abs_cos_angle)  # [*]
    angle_deg = torch.rad2deg(angle_rad)  # [*]

    return angle_deg


@torch.no_grad()
def eval_antipode_angle(p: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    Antipode-aware angle between prediction and solution (EVALUATION version).

    Uses clamp(-1.0, 1.0) so the full [0°, 90°] range is representable.
    Must only be used for metric computation (no backprop).

    Args:
        p: [*, 3] prediction (assumed unit vectors)
        s: [*, 3] solution (assumed unit vectors)

    Returns:
        [*] angles in degrees, accounting for antipodal symmetry
    """
    cos_angle = torch.sum(p * s, dim=-1)
    abs_cos = torch.abs(cos_angle)
    abs_cos = torch.clamp(abs_cos, 0.0, 1.0)
    angle_rad = torch.acos(abs_cos)
    return torch.rad2deg(angle_rad)


def oracle_k3_loss(
    poles: torch.Tensor,
    solutions_padded: Optional[torch.Tensor] = None,
    solutions_mask: Optional[torch.Tensor] = None,
    solutions_list: Optional[List[Optional[torch.Tensor]]] = None,
    tau_deg: float = 5.0,
) -> torch.Tensor:
    """
    Oracle@3 loss: best of K=3 hypotheses.

    Supports two input formats:
    1. Vectorized (preferred): solutions_padded [B, max_sols, 3] + solutions_mask [B, max_sols]
    2. Legacy list-based: solutions_list (slower, for compatibility)

    Args:
        poles: [B, K=3, 3] predicted poles
        solutions_padded: [B, max_sols, 3] padded solution tensor
        solutions_mask: [B, max_sols] boolean mask for valid solutions
        solutions_list: legacy Python list of tensors (for compatibility)
        tau_deg: softmin temperature in degrees

    Returns:
        scalar loss (mean angle error across batch)
    """
    B, K, _ = poles.shape
    assert K == 3, f"Expected K=3, got K={K}"

    # Use vectorized path if available
    if solutions_padded is not None and solutions_mask is not None:
        _, max_sols, _ = solutions_padded.shape
        tau = float(tau_deg)
        if tau <= 0:
            tau = 1.0

        # Expand poles to [B, K, max_sols, 3]
        poles_expanded = poles.unsqueeze(2).expand(B, K, max_sols, 3)

        # Expand solutions to [B, K, max_sols, 3]
        sols_expanded = solutions_padded.unsqueeze(1).expand(B, K, max_sols, 3)

        # Flatten for antipode_angle computation
        poles_flat = poles_expanded.reshape(B * K * max_sols, 3)
        sols_flat = sols_expanded.reshape(B * K * max_sols, 3)

        # Compute angles
        angles_flat = antipode_angle(poles_flat, sols_flat)
        angles = angles_flat.reshape(B, K, max_sols)

        # Apply mask
        mask_expanded = solutions_mask.unsqueeze(1).expand(B, K, max_sols)
        angles = angles.masked_fill(~mask_expanded, float("inf"))

        # Softmin over solutions (per pole): [B, K]
        per_pole = -tau * torch.logsumexp(-angles / tau, dim=2)

        # Softmin over K hypotheses (oracle@K): [B]
        oracle_losses = -tau * torch.logsumexp(-per_pole / tau, dim=1)

        # Filter out batch items with no valid solutions.
        valid_mask = solutions_mask.any(dim=1)
        if valid_mask.sum() == 0:
            # Return a zero loss connected to the computation graph
            return (poles * 0).sum()

        return oracle_losses[valid_mask].mean()

    # Legacy list-based path (slower but maintains backward compatibility)
    elif solutions_list is not None:
        losses = []

        for b in range(B):
            if solutions_list[b] is None or len(solutions_list[b]) == 0:
                continue

            sols = solutions_list[b]  # [S, 3]
            p0 = poles[b, 0]  # [3]
            p1 = poles[b, 1]  # [3]
            p2 = poles[b, 2]  # [3]

            # Min angle to any solution for each pole
            angles_p0 = antipode_angle(p0.unsqueeze(0).expand(len(sols), -1), sols)
            angles_p1 = antipode_angle(p1.unsqueeze(0).expand(len(sols), -1), sols)
            angles_p2 = antipode_angle(p2.unsqueeze(0).expand(len(sols), -1), sols)

            min_angle_p0 = angles_p0.min()
            min_angle_p1 = angles_p1.min()
            min_angle_p2 = angles_p2.min()

            # Oracle@3 loss: best of all 3 poles
            oracle_loss = torch.minimum(torch.minimum(min_angle_p0, min_angle_p1), min_angle_p2)

            losses.append(oracle_loss)

        if losses:
            return torch.mean(torch.stack(losses))
        else:
            # Return a zero loss connected to the computation graph
            return (poles * 0).sum()
    else:
        raise ValueError("Must provide either (solutions_padded, solutions_mask) or solutions_list")


def gap_weighted_quality_loss_k3(
    quality_logits: torch.Tensor,
    poles: torch.Tensor,
    solutions_list: List[Optional[torch.Tensor]],
    gap_tau_deg: float = 10.0,
    ignore_near_ties_deg: float = 1.0,
    curriculum_epoch: Optional[int] = None,
    curriculum_max_epochs: int = 200,
) -> torch.Tensor:
    """
    Gap-weighted quality head loss (K=3 version) with optional curriculum learning.

    For K=3, use one-vs-rest gap weighting:
    - Compare each pole against minimum error from the other two
    - Weight by gap to encourage selection of better poles

    Args:
        quality_logits: [B, 3] logits for quality head
        poles: [B, 3, 3] - 3 pole predictions per sample
        solutions_list: list of [S, 3] solution tensors
        gap_tau_deg: scaling for gap weighting
        ignore_near_ties_deg: threshold below which skip supervision
        curriculum_epoch: current epoch (for curriculum schedule). If None, use standard loss.
        curriculum_max_epochs: total number of epochs for curriculum

    Returns:
        scalar loss
    """
    B = poles.shape[0]
    device = poles.device

    # Compute curriculum threshold for gap
    if curriculum_epoch is not None:
        # Schedule: linearly decrease from 5° to 0.5° over training
        max_gap_threshold = 5.0
        min_gap_threshold = 0.5
        progress = curriculum_epoch / curriculum_max_epochs
        gap_threshold = max_gap_threshold - progress * (max_gap_threshold - min_gap_threshold)
    else:
        gap_threshold = ignore_near_ties_deg

    # Compute losses for all batch items
    loss_per_batch = []
    weight_per_batch = []

    for b in range(B):
        if solutions_list[b] is None or len(solutions_list[b]) == 0:
            continue

        sols = solutions_list[b]  # [S, 3]
        p0 = poles[b, 0]  # [3]
        p1 = poles[b, 1]  # [3]
        p2 = poles[b, 2]  # [3]

        # Compute angles for all 3 poles
        angles_p0 = antipode_angle(p0.unsqueeze(0).expand(len(sols), -1), sols)
        angles_p1 = antipode_angle(p1.unsqueeze(0).expand(len(sols), -1), sols)
        angles_p2 = antipode_angle(p2.unsqueeze(0).expand(len(sols), -1), sols)

        # Minimum angles for each pole
        min_angle_p0 = angles_p0.min()
        min_angle_p1 = angles_p1.min()
        min_angle_p2 = angles_p2.min()

        # Create target: best pole index
        min_angles = torch.stack([min_angle_p0, min_angle_p1, min_angle_p2])
        best_idx = torch.argmin(min_angles)  # Scalar tensor with value 0, 1, or 2

        # Compute weighted gap loss
        # For each pole i, gap = min(errors of other poles) - min(error of pole i)
        gaps = []
        for i in range(3):
            # Get all angles except for pole i
            other_angles = min_angles[[j for j in range(3) if j != i]]
            gap = torch.clamp(other_angles.min() - min_angles[i], min=0.0)
            gaps.append(gap)

        gaps_tensor = torch.stack(gaps)  # [3]
        avg_gap = gaps_tensor.mean()

        # Skip if average gap too small
        if avg_gap.item() < gap_threshold:
            continue

        # Weight by average gap
        weight = torch.clamp(avg_gap / gap_tau_deg, max=1.0)

        # Compute CE loss for this batch item
        # Target should be class index (0, 1, or 2), not one-hot
        item_loss = F.cross_entropy(
            quality_logits[b:b+1],  # [1, 3]
            best_idx.unsqueeze(0),  # [1] with class index
        )

        loss_per_batch.append(item_loss)
        weight_per_batch.append(weight)

    if loss_per_batch:
        # Stack losses and weights
        losses_stacked = torch.stack(loss_per_batch)  # [M]
        weights_stacked = torch.stack(weight_per_batch)  # [M]

        # Weighted mean
        loss = (losses_stacked * weights_stacked).sum() / weights_stacked.sum()
        return loss
    else:
        # Return zero loss connected to computation graph
        return (quality_logits * 0).sum()


# ============================================================================
# ANTI-COLLAPSE LOSSES
# ============================================================================

def batch_variance_loss(poles: torch.Tensor) -> torch.Tensor:
    """
    Penalize identical predictions across a batch by maximizing
    per-slot angular variance.

    For each slot k, compute mean pairwise |cosine similarity| across batch
    samples. If all samples predict the same pole, this is 1.0. If predictions
    are diverse, this approaches 0.

    Args:
        poles: [B, K, 3] predicted unit vectors

    Returns:
        scalar loss (0 = maximally diverse, 1 = all identical)
    """
    B, K, D = poles.shape
    if B < 2:
        return torch.tensor(0.0, device=poles.device, requires_grad=True)

    total = torch.tensor(0.0, device=poles.device)

    for k in range(K):
        p = poles[:, k]  # [B, 3]
        cos_sim = torch.mm(p, p.T)
        abs_cos_sim = cos_sim.abs()
        mask = ~torch.eye(B, dtype=torch.bool, device=poles.device)
        off_diag = abs_cos_sim[mask]
        total = total + off_diag.mean()

    return total / K


def similarity_matching_loss(
    poles: torch.Tensor,
    solutions_list: List[Optional[torch.Tensor]],
) -> torch.Tensor:
    """
    Similarity-matching loss: prediction similarity should match GT similarity.

    Samples with similar GT poles should predict similar poles, and vice versa.
    Uses the first GT solution per sample as the reference.

    Args:
        poles: [B, K, 3] predicted poles
        solutions_list: list of [S, 3] solution tensors

    Returns:
        scalar loss
    """
    B, K, D = poles.shape
    device = poles.device

    if B < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    gt_poles = []
    valid_mask = []
    for b in range(B):
        if solutions_list[b] is not None and len(solutions_list[b]) > 0:
            gt_poles.append(F.normalize(solutions_list[b][0:1], dim=-1).squeeze(0))
            valid_mask.append(True)
        else:
            gt_poles.append(torch.zeros(3, device=device))
            valid_mask.append(False)

    gt = torch.stack(gt_poles)
    valid = torch.tensor(valid_mask, device=device)

    if valid.sum() < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    gt_sim = torch.mm(gt, gt.T).abs()
    pair_valid = valid.unsqueeze(0).float() * valid.unsqueeze(1).float()
    off_diag = ~torch.eye(B, dtype=torch.bool, device=device)
    pair_valid = pair_valid * off_diag.float()

    if pair_valid.sum() < 1:
        return torch.tensor(0.0, device=device, requires_grad=True)

    total = torch.tensor(0.0, device=device)

    for k in range(K):
        p = poles[:, k]
        pred_sim = torch.mm(p, p.T).abs()
        diff = (pred_sim - gt_sim) ** 2
        loss_k = (diff * pair_valid).sum() / pair_valid.sum().clamp(min=1)
        total = total + loss_k

    return total / K


def continuous_diversity_loss(
    poles: torch.Tensor,
    sigma_deg: float = 15.0,
) -> torch.Tensor:
    """
    Continuous diversity loss with exponential repulsion.

    Unlike hinge diversity (ReLU(margin - angle)), this always has gradient.
    Poles that are close get strongly repelled; far poles get weak repulsion.

    L = mean(exp(-angle_ij / sigma)) for all pairs i < j

    Args:
        poles: [B, K, 3] predicted poles
        sigma_deg: temperature in degrees (controls repulsion range)

    Returns:
        scalar loss
    """
    B, K, D = poles.shape

    pair_losses = []
    for i in range(K):
        for j in range(i + 1, K):
            angles = antipode_angle(poles[:, i], poles[:, j])
            repulsion = torch.exp(-angles / sigma_deg)
            pair_losses.append(repulsion.mean())

    if not pair_losses:
        return torch.tensor(0.0, device=poles.device, requires_grad=True)

    return torch.stack(pair_losses).mean()


def combined_loss(
    poles: torch.Tensor,
    quality_logits: Optional[torch.Tensor],
    solutions_list: List[Optional[torch.Tensor]],
    *,
    lambda_div: float = 0.5,
    lambda_q: float = 0.0,
    lambda_var: float = 5.0,
    lambda_sim: float = 2.0,
    softmin_tau_deg: float = 5.0,
    div_sigma_deg: float = 15.0,
    gap_tau_deg: float = 10.0,
    ignore_near_ties_deg: float = 1.0,
) -> dict:
    """
    Combined loss with anti-collapse terms.

    Args:
        poles: [B, K, 3] predicted poles
        quality_logits: [B, K] or None
        solutions_list: list of solution tensors
        lambda_div: weight for continuous diversity loss
        lambda_q: weight for quality loss
        lambda_var: weight for batch variance loss
        lambda_sim: weight for similarity-matching loss
        softmin_tau_deg: oracle softmin temperature
        div_sigma_deg: diversity sigma (degrees)
        gap_tau_deg: quality loss gap tau
        ignore_near_ties_deg: quality loss near-tie threshold

    Returns:
        dict with 'loss' (total), 'L_pole', 'L_div', 'L_var', 'L_sim', 'L_q'
    """
    solutions_padded, solutions_mask = vectorize_solutions(
        solutions_list, device=poles.device
    )

    L_pole = oracle_k3_loss(
        poles,
        solutions_padded=solutions_padded,
        solutions_mask=solutions_mask,
        tau_deg=softmin_tau_deg,
    )

    L_div = continuous_diversity_loss(poles, sigma_deg=div_sigma_deg)
    L_var = batch_variance_loss(poles)

    L_sim = torch.tensor(0.0, device=poles.device)
    if lambda_sim > 0:
        L_sim = similarity_matching_loss(poles, solutions_list)

    L_q = torch.tensor(0.0, device=poles.device)
    if lambda_q > 0 and quality_logits is not None:
        L_q = gap_weighted_quality_loss_k3(
            quality_logits, poles, solutions_list,
            gap_tau_deg=gap_tau_deg,
            ignore_near_ties_deg=ignore_near_ties_deg,
        )

    loss = (
        L_pole
        + lambda_div * L_div
        + lambda_var * L_var
        + lambda_sim * L_sim
        + lambda_q * L_q
    )

    return {
        'loss': loss,
        'L_pole': L_pole.item(),
        'L_div': L_div.item(),
        'L_var': L_var.item(),
        'L_sim': L_sim.item() if lambda_sim > 0 else 0.0,
        'L_q': L_q.item() if lambda_q > 0 else 0.0,
    }


# Backward-compatibility aliases for external scripts
combined_loss_v2 = combined_loss
combined_loss_k3 = combined_loss


