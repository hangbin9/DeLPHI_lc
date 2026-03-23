"""
Transformer for asteroid pole prediction.

Architecture:
  1. Token projection + LayerNorm
  2. Window-level transformer encoder (GELU activation)
  3. Attention pooling (per-window, then mean across windows)
  4. K=3 pole prediction heads
"""
import logging
from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class WindowTransformer(nn.Module):
    """Transformer encoder for tokens within a single window."""

    def __init__(self, d_model: int, n_heads: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)


class AttentionPooling(nn.Module):
    """Attention-based pooling over token sequence to [d_model]."""

    def __init__(self, d_model: int):
        super().__init__()
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.pool_query, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            mask: [B, T], 1 = valid, 0 = pad
        Returns:
            [B, d_model]
        """
        query = self.pool_query.expand(x.size(0), -1, -1)  # [B, 1, d_model]
        scores = torch.matmul(query, x.transpose(1, 2)) / (self.pool_query.size(-1) ** 0.5)
        scores = scores + (1 - mask).unsqueeze(1) * (-1e9)
        attn = F.softmax(scores, dim=-1)
        pooled = torch.matmul(attn, x)
        return pooled.squeeze(1)


class K3SlotHead(nn.Module):
    """Independent MLP head for one slot (pole + optional quality)."""

    def __init__(self, d_model: int, include_quality: bool = False):
        super().__init__()
        self.include_quality = include_quality

        self.pole_mlp = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

        if include_quality:
            self.quality_mlp = nn.Sequential(
                nn.Linear(d_model, 2 * d_model),
                nn.GELU(),
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        pole_raw = self.pole_mlp(z)
        poles = F.normalize(pole_raw, p=2, dim=-1)

        quality = None
        if self.include_quality:
            quality = self.quality_mlp(z)

        return poles, quality


class PolePredictor(nn.Module):
    """
    Transformer for pole prediction.

    Input: tokens [B, W, T, F], mask [B, W, T]
    Output: poles [B, 3, 3], quality [B, 3] or None

    Architecture:
      token_proj (n_features -> d_model) -> LayerNorm
      -> encoder (4 layers, GELU) -> attention_pool -> z [B, d_model]
      -> 3 x K3SlotHead -> poles [B, 3, 3]

    Multi-window inputs (W > 1) are handled by mean-pooling
    per-window embeddings.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_features: int = 13,
        dropout: float = 0.1,
        include_quality_head: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.include_quality_head = include_quality_head

        # Token projection
        self.token_proj = nn.Linear(n_features, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Single encoder (no cross-window path)
        self.encoder = WindowTransformer(d_model, n_heads, n_layers, dropout)
        self.pool = AttentionPooling(d_model)

        # K=3 slot heads
        self.slot_head_0 = K3SlotHead(d_model, include_quality=include_quality_head)
        self.slot_head_1 = K3SlotHead(d_model, include_quality=include_quality_head)
        self.slot_head_2 = K3SlotHead(d_model, include_quality=include_quality_head)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tokens: [B, W, T, F] token sequences
            mask: [B, W, T] padding mask (1 = valid)

        Returns:
            (poles [B, 3, 3], quality_logits [B, 3] or None)
        """
        B, W, T, F_dim = tokens.shape

        # Flatten windows into batch
        tokens_flat = tokens.view(B * W, T, F_dim)
        mask_flat = mask.view(B * W, T)

        # Project and normalize
        projected = self.token_proj(tokens_flat)  # [B*W, T, d_model]
        projected = self.input_norm(projected)

        # Encode
        key_pad_mask = (1 - mask_flat).bool()
        encoded = self.encoder(projected, key_pad_mask)  # [B*W, T, d_model]

        # Pool tokens -> single embedding per window
        pooled = self.pool(encoded, mask_flat)  # [B*W, d_model]

        # Reshape and mean-pool across windows
        pooled = pooled.view(B, W, self.d_model)
        z = pooled.mean(dim=1)  # [B, d_model]

        # Slot heads
        poles_0, q0 = self.slot_head_0(z)
        poles_1, q1 = self.slot_head_1(z)
        poles_2, q2 = self.slot_head_2(z)

        poles = torch.stack([poles_0, poles_1, poles_2], dim=1)

        quality_logits = None
        if self.include_quality_head:
            quality_logits = torch.cat([q0, q1, q2], dim=1)

        return poles, quality_logits

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cpu") -> "PolePredictor":
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        config = ckpt.get('config', {})
        state_dict = ckpt['model_state_dict']

        # Detect quality head from state dict
        has_quality = any('quality_mlp' in k for k in state_dict.keys())
        include_quality_head = config.get('include_quality_head', has_quality)

        # Get number of input features from checkpoint
        n_features = state_dict.get('token_proj.weight', torch.zeros(128, 13)).shape[1]

        model = cls(
            d_model=config.get('d_model', 128),
            n_heads=config.get('n_heads', 4),
            n_layers=config.get('n_layers', 4),
            n_features=n_features,
            include_quality_head=include_quality_head,
        )

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            if 'Unexpected key' in str(e) or 'Missing key' in str(e):
                model_state = model.state_dict()
                filtered_state = {k: v for k, v in state_dict.items() if k in model_state}
                model.load_state_dict(filtered_state, strict=False)
                logger.warning(f"Loaded checkpoint with partial state dict (filtered {len(state_dict) - len(filtered_state)} keys)")
            else:
                raise

        model = model.to(device)
        model.eval()
        return model
