"""
Factored MoE Projection layer — replaces a single 1x1 Conv2d in MBConv blocks.

Each expert is a bottleneck: Conv1x1(C_in -> r) -> SiLU -> Conv1x1(r -> C_out) -> BN.
Only top-k experts are activated per image, reducing inference cost.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gating import NoisyTopKGate


class FactoredMoEProjection(nn.Module):
    """
    Replaces a Conv2d(C_in, C_out, 1x1) with N factored experts + gating.

    Each expert: C_in -> r -> C_out (bottleneck)
    At inference, only top-k experts run per image.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 1,
        bottleneck_rank: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.top_k = top_k
        self.bottleneck_rank = bottleneck_rank

        # Gate: routes based on globally-pooled features
        self.gate = NoisyTopKGate(in_channels, num_experts, top_k)

        # Experts: each is a bottleneck 1x1 conv pair + BN
        self.experts_down = nn.ModuleList([
            nn.Conv2d(in_channels, bottleneck_rank, 1, bias=False)
            for _ in range(num_experts)
        ])
        self.experts_up = nn.ModuleList([
            nn.Conv2d(bottleneck_rank, out_channels, 1, bias=False)
            for _ in range(num_experts)
        ])
        self.experts_bn = nn.ModuleList([
            nn.BatchNorm2d(out_channels)
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C_in, H, W)

        Returns:
            output: (B, C_out, H, W)
            aux_losses: dict with load_balance_loss and z_loss
        """
        B, C, H, W = x.shape

        # Global average pool for gating decision
        x_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, C_in)

        # Get routing weights and indices
        expert_weights, expert_indices, aux_losses = self.gate(x_pool)
        # expert_weights: (B, top_k), expert_indices: (B, top_k)

        # Accumulate in float32 for AMP safety (BN outputs float32 under autocast)
        output = torch.zeros(B, self.out_channels, H, W, device=x.device)

        for k in range(self.top_k):
            weight_k = expert_weights[:, k]  # (B,)
            indices_k = expert_indices[:, k]  # (B,)

            # Process each expert's assigned batch
            for e in range(self.num_experts):
                mask = (indices_k == e)  # which images go to expert e
                if not mask.any():
                    continue

                x_e = x[mask]  # (n_e, C_in, H, W)
                # Bottleneck: down -> SiLU -> up -> BN
                h = self.experts_down[e](x_e)
                h = F.silu(h)
                h = self.experts_up[e](h)
                h = self.experts_bn[e](h)

                # Weighted contribution
                w = weight_k[mask].view(-1, 1, 1, 1)  # (n_e, 1, 1, 1)
                output[mask] = output[mask] + h * w

        return output, aux_losses

    def count_active_params(self) -> int:
        """Count parameters activated per sample (top-k experts + gate)."""
        gate_params = sum(p.numel() for p in self.gate.parameters())
        # Only w_gate counts at inference (w_noise is training only)
        gate_inference_params = (
            self.gate.w_gate.weight.numel() + self.gate.w_gate.bias.numel()
        )
        single_expert_params = (
            self.experts_down[0].weight.numel()
            + self.experts_up[0].weight.numel()
            + self.experts_bn[0].weight.numel()
            + self.experts_bn[0].bias.numel()
        )
        return gate_inference_params + self.top_k * single_expert_params
