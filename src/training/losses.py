"""
Training losses for EfficientNet-MoE.

Total loss = L_CE + alpha * L_balance + beta * L_z

- L_CE: cross-entropy with label smoothing (handles class imbalance)
- L_balance: load-balancing loss (prevents expert collapse)
- L_z: router z-loss (stabilizes gate logits)
"""

import torch
import torch.nn as nn


class MoELoss(nn.Module):

    def __init__(self, num_classes: int = 1010, label_smoothing: float = 0.1,
                 alpha: float = 0.01, beta: float = 0.001):
        """
        Args:
            num_classes: number of species classes
            label_smoothing: label smoothing factor for CE loss
            alpha: weight for load-balancing loss
            beta: weight for router z-loss
        """
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets, aux_losses):
        """
        Args:
            logits: (B, num_classes) model output
            targets: (B,) species labels
            aux_losses: dict with 'load_balance_loss' and 'z_loss' from model

        Returns:
            total_loss: scalar
            loss_dict: breakdown for logging
        """
        ce = self.ce_loss(logits, targets)
        balance = aux_losses["load_balance_loss"]
        z = aux_losses["z_loss"]

        total = ce + self.alpha * balance + self.beta * z

        return total, {
            "total": total.item(),
            "ce": ce.item(),
            "balance": balance.item(),
            "z": z.item(),
        }
