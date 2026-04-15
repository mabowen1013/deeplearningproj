"""
EfficientNet-B0 with Sparse MoE — main model assembly.

Loads pretrained EfficientNet-B0 and surgically replaces
two late projection layers with factored sparse MoE projections, 
inserts an MoEFFNBlock between Stage 6 and Stage 7,
and replaces the classifier head for iNaturalist 2019 (1010 classes).
"""

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from .moe_layer import FactoredMoEProjection
from .moe_ffn import MoEFFNBlock


class MoEWrapper(nn.Module):
    """
    Wraps FactoredMoEProjection so it fits inside a Sequential.

    The MBConv block uses Sequential for its sub-layers, which expects
    each module to return a single tensor. This wrapper stores aux_losses
    as an attribute and returns only the tensor.
    """

    def __init__(self, moe_layer: FactoredMoEProjection):
        super().__init__()
        self.moe = moe_layer
        self.aux_losses = {}

    def forward(self, x):
        out, self.aux_losses = self.moe(x)
        return out

    def count_active_params(self):
        return self.moe.count_active_params()


# Which MBConv blocks to convert: (stage_idx, block_idx, in_ch, out_ch, rank)
# Only Stage 6-7 where routing is meaningful (NMI > 0.08).
# Stage 5 removed: NMI ~0.03-0.04, features are shared across taxa.
# Rank increased from 48 to 64 to use more of the parameter budget.
MOE_TARGETS = [
    # Stage 6: features[6], block 1 (highest proj NMI after S7)
    (6, 1, 1152, 192, 64),
    # Stage 7: features[7], block 0 (second highest proj NMI)
    (7, 0, 1152, 320, 64),
]


class EfficientNetMoE(nn.Module):

    def __init__(
        self,
        num_classes: int = 1010,
        num_experts_proj: int = 4,
        top_k_proj: int = 1,
        num_experts_ffn: int = 8,
        top_k_ffn: int = 2,
        ffn_hidden_dim: int = 192,
        pretrained: bool = True,
    ):
        super().__init__()

        # Load pretrained EfficientNet-B0
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = efficientnet_b0(weights=weights)

        # Split features into before/after Stage 6 for FFN block insertion
        # features[0..6] = Stem + Stage1-6, features[7] = Stage7, features[8] = Final conv
        self.features_before = nn.Sequential(*list(base.features[:7]))  # [0] to [6]
        self.moe_ffn = MoEFFNBlock(
            dim=192, hidden_dim=ffn_hidden_dim,
            num_experts=num_experts_ffn, top_k=top_k_ffn,
        )
        self.features_after = nn.Sequential(*list(base.features[7:]))   # [7] and [8]

        self.avgpool = base.avgpool  # AdaptiveAvgPool2d(1)

        # Replace classifier for iNaturalist (1010 species)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1280, num_classes),
        )

        # Replace projection layers with MoE
        self.moe_wrappers = []  # track for aux loss collection
        for stage_idx, block_idx, in_ch, out_ch, rank in MOE_TARGETS:
            moe = FactoredMoEProjection(
                in_channels=in_ch,
                out_channels=out_ch,
                num_experts=num_experts_proj,
                top_k=top_k_proj,
                bottleneck_rank=rank,
            )
            wrapper = MoEWrapper(moe)
            self.moe_wrappers.append(wrapper)

            # Navigate to the right block and replace block[3]
            if stage_idx < 7:
                mbconv = self.features_before[stage_idx][block_idx]
            else:
                # features_after[0] = original features[7]
                mbconv = self.features_after[0][block_idx]

            mbconv.block[3] = wrapper

        # Register moe_wrappers as a ModuleList so params are tracked
        self.moe_wrappers = nn.ModuleList(self.moe_wrappers)

    def forward(self, x):
        """
        Returns:
            logits: (B, num_classes)
            aux_losses: dict with aggregated load_balance_loss and z_loss
        """
        # Features before MoE FFN (Stem + Stage 1-6)
        x = self.features_before(x)
        # x: (B, 192, 7, 7)

        # MoE FFN block
        x, ffn_aux = self.moe_ffn(x)

        # Features after (Stage 7 + Final conv)
        x = self.features_after(x)
        # x: (B, 1280, 7, 7)

        # Pool and classify
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # Collect all auxiliary losses
        aux_losses = self._collect_aux_losses(ffn_aux)

        return logits, aux_losses

    def _collect_aux_losses(self, ffn_aux):
        """Aggregate aux losses from all MoE layers."""
        total_balance = ffn_aux["load_balance_loss"]
        total_z = ffn_aux["z_loss"]
        n_layers = 1  # start with FFN block

        for wrapper in self.moe_wrappers:
            total_balance = total_balance + wrapper.aux_losses["load_balance_loss"]
            total_z = total_z + wrapper.aux_losses["z_loss"]
            n_layers += 1

        # Average over layers
        return {
            "load_balance_loss": total_balance / n_layers,
            "z_loss": total_z / n_layers,
        }

    def get_param_stats(self):
        """Return total and active parameter counts."""
        total = sum(p.numel() for p in self.parameters())

        # Active = total - inactive expert params
        # Inactive = (num_experts - top_k) * single_expert_params per MoE layer
        inactive = 0
        for wrapper in self.moe_wrappers:
            moe = wrapper.moe
            single = (
                moe.experts_down[0].weight.numel()
                + moe.experts_up[0].weight.numel()
                + moe.experts_bn[0].weight.numel()
                + moe.experts_bn[0].bias.numel()
            )
            inactive += (moe.num_experts - moe.top_k) * single
            # w_noise is training-only
            inactive += moe.gate.w_noise.weight.numel()

        # MoE FFN block
        ffn = self.moe_ffn
        single_ffn = (
            ffn.experts_fc1[0].weight.numel() + ffn.experts_fc1[0].bias.numel()
            + ffn.experts_fc2[0].weight.numel() + ffn.experts_fc2[0].bias.numel()
        )
        inactive += (ffn.num_experts - ffn.top_k) * single_ffn
        inactive += ffn.gate.w_noise.weight.numel()

        active = total - inactive
        return {"total": total, "active": active, "inactive": inactive}
