"""
Cosine annealing learning rate scheduler with linear warmup.
"""

import math
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_with_warmup(optimizer, warmup_epochs: int, total_epochs: int):
    """
    Linear warmup for warmup_epochs, then cosine decay to 0.

    Args:
        optimizer: the optimizer
        warmup_epochs: number of warmup epochs
        total_epochs: total training epochs
    """

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: 0 -> 1
            return epoch / max(1, warmup_epochs)
        # Cosine decay: 1 -> 0
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)
