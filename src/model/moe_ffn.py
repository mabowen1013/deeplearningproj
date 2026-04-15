"""
Independent MoE FFN Block — inserted between Stage 6 and Stage 7.

Provides an explicit point of image-level specialization with more experts (8),
making taxonomy analysis more interpretable. Uses top-2 routing.

Structure: x -> LayerNorm -> MoE(experts) -> residual add
Each expert: Linear(dim, hidden) -> SiLU -> Linear(hidden, dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating import NoisyTopKGate


class MoEFFNBlock(nn.Module):
    """
    MoE Feed-Forward block operating on spatially-pooled features,
    then broadcasting back to the spatial dimensions.

    Input:  (B, C, H, W)
    Output: (B, C, H, W)  (same shape, residual connection)
    """

    def __init__(
        self,
        dim: int = 192,
        hidden_dim: int = 128,
        num_experts: int = 8,
        top_k: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k

        self.norm = nn.LayerNorm(dim)
        self.gate = NoisyTopKGate(dim, num_experts, top_k)

        # Each expert: Linear -> SiLU -> Linear
        self.experts_fc1 = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for _ in range(num_experts)
        ])
        self.experts_fc2 = nn.ModuleList([
            nn.Linear(hidden_dim, dim) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            output: (B, C, H, W) — x + MoE(LayerNorm(pool(x))) broadcast
            aux_losses: dict
        """
        B, C, H, W = x.shape

        # Global average pool -> (B, C)
        x_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)

        # LayerNorm on pooled features
        x_norm = self.norm(x_pool)  # (B, C)

        # Gating
        expert_weights, expert_indices, aux_losses = self.gate(x_norm)
        # expert_weights: (B, top_k), expert_indices: (B, top_k)

        # Accumulate in float32 for AMP safety
        ffn_out = torch.zeros(B, self.dim, device=x.device)

        for k in range(self.top_k):
            weight_k = expert_weights[:, k]  # (B,)
            indices_k = expert_indices[:, k]  # (B,)

            for e in range(self.num_experts):
                mask = (indices_k == e)
                if not mask.any():
                    continue

                h = x_norm[mask]  # (n_e, dim)
                h = self.experts_fc1[e](h)
                h = F.silu(h)
                h = self.experts_fc2[e](h)

                w = weight_k[mask].unsqueeze(-1)  # (n_e, 1)
                ffn_out[mask] = ffn_out[mask] + h * w

        # Broadcast pooled MoE output back to spatial dims and add residual
        output = x + ffn_out.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1) broadcast

        return output, aux_losses

    def count_active_params(self) -> int:
        """Count parameters activated per sample at inference."""
        gate_params = self.gate.w_gate.weight.numel() + self.gate.w_gate.bias.numel()
        norm_params = self.norm.weight.numel() + self.norm.bias.numel()
        single_expert_params = (
            self.experts_fc1[0].weight.numel() + self.experts_fc1[0].bias.numel()
            + self.experts_fc2[0].weight.numel() + self.experts_fc2[0].bias.numel()
        )
        return gate_params + norm_params + self.top_k * single_expert_params
