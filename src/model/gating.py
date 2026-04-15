"""
Noisy Top-K Gating for Sparse Mixture of Experts.

References:
  - Shazeer et al., "Outrageously Large Neural Networks" (ICLR 2017)
  - Fedus et al., "Switch Transformers" (JMLR 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyTopKGate(nn.Module):
    """
    Per-image gating network that routes each input to top-k experts.

    Given a pooled feature vector, produces:
      - expert indices (which experts to activate)
      - expert weights (softmax weights for combining expert outputs)
      - auxiliary losses (load balancing + z-loss)
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Gate projection: input_dim -> num_experts
        self.w_gate = nn.Linear(input_dim, num_experts, bias=True)

        # Learnable noise for exploration during training (Shazeer et al.)
        self.w_noise = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, input_dim) — pooled feature vector per image

        Returns:
            expert_weights: (B, top_k) — softmax weights for selected experts
            expert_indices: (B, top_k) — indices of selected experts
            aux_losses: dict with 'load_balance_loss' and 'z_loss' scalars
        """
        # Gate logits: (B, num_experts)
        logits = self.w_gate(x)

        # Add noise during training to encourage exploration
        if self.training:
            noise_stddev = F.softplus(self.w_noise(x))  # (B, num_experts)
            noise = torch.randn_like(logits) * noise_stddev
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        # Top-k selection
        top_k_values, top_k_indices = torch.topk(noisy_logits, self.top_k, dim=-1)
        # (B, top_k), (B, top_k)

        # Softmax over selected experts only
        top_k_weights = F.softmax(top_k_values, dim=-1)  # (B, top_k)

        # Compute auxiliary losses
        aux_losses = self._compute_aux_losses(logits, noisy_logits)

        return top_k_weights, top_k_indices, aux_losses

    def _compute_aux_losses(self, logits, noisy_logits):
        """
        Compute load-balancing loss and router z-loss.

        Load-balancing loss (Switch Transformer):
            L_balance = N * sum_i(f_i * p_i)
            where f_i = fraction of inputs routed to expert i
                  p_i = mean gate probability for expert i

        Z-loss (ST-MoE):
            L_z = mean((logsumexp(logits))^2)
            Penalizes large logits for training stability.
        """
        B = logits.shape[0]
        num_experts = self.num_experts

        # f_i: fraction of inputs dispatched to each expert
        # Use noisy_logits for the dispatch decision (matches actual routing)
        dispatch_indices = torch.argmax(noisy_logits, dim=-1)  # (B,)
        # One-hot: (B, num_experts), then average over batch
        f = torch.zeros(B, num_experts, device=logits.device)
        f.scatter_(1, dispatch_indices.unsqueeze(1), 1.0)
        f = f.mean(dim=0)  # (num_experts,)

        # p_i: mean gate probability (from clean logits, differentiable)
        p = F.softmax(logits, dim=-1).mean(dim=0)  # (num_experts,)

        # Load balancing loss
        load_balance_loss = num_experts * (f * p).sum()

        # Router z-loss: penalize large logits
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        return {
            "load_balance_loss": load_balance_loss,
            "z_loss": z_loss,
        }
