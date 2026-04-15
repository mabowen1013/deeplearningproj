"""
Expert routing data collection.

Runs the full val set through the model, recording gate decisions
at each MoE layer for every image. Output is used by taxonomy_viz.py.
"""

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader
from collections import Counter, defaultdict

from ..model.efficientnet_moe import EfficientNetMoE
from ..data.dataset import INat2019Dataset


@torch.no_grad()
def collect_routing_data(model: EfficientNetMoE, val_loader: DataLoader, device):
    """
    Collect per-image expert routing decisions across all MoE layers.

    Returns:
        routing_data: list of dicts, one per image:
            {
                species: int,
                super_cat: int,
                proj_experts: [int] * 6,       # top-1 expert per projection MoE
                proj_gate_probs: [Tensor] * 6,  # full gate probabilities (N experts)
                ffn_experts: [int] * 2,         # top-2 experts for FFN MoE
                ffn_gate_probs: Tensor,          # full gate probabilities (8 experts)
            }
    """
    model.eval()

    # Hook into gates to capture full gate probabilities (before top-k masking)
    gate_data = {}

    def make_gate_hook(name):
        def hook(module, input, output):
            weights, indices, aux = output
            # Also capture the raw softmax probs from the clean logits
            x = input[0]  # (B, input_dim)
            with torch.no_grad():
                clean_logits = module.w_gate(x)
                probs = torch.softmax(clean_logits, dim=-1)  # (B, N)
            gate_data[name] = {
                "indices": indices.cpu(),
                "probs": probs.cpu(),
            }
        return hook

    hooks = []
    for i, wrapper in enumerate(model.moe_wrappers):
        h = wrapper.moe.gate.register_forward_hook(make_gate_hook(f"proj_{i}"))
        hooks.append(h)
    h = model.moe_ffn.gate.register_forward_hook(make_gate_hook("ffn"))
    hooks.append(h)

    routing_data = []

    for images, species, super_cats in val_loader:
        images = images.to(device, non_blocking=True)
        gate_data.clear()

        with autocast("cuda"):
            model(images)

        B = images.size(0)
        for b in range(B):
            entry = {
                "species": species[b].item(),
                "super_cat": super_cats[b].item(),
                "proj_experts": [],
                "proj_gate_probs": [],
                "ffn_experts": [],
                "ffn_gate_probs": None,
            }
            for i in range(len(model.moe_wrappers)):
                gd = gate_data[f"proj_{i}"]
                entry["proj_experts"].append(gd["indices"][b].item())
                entry["proj_gate_probs"].append(gd["probs"][b])

            gd_ffn = gate_data["ffn"]
            entry["ffn_experts"] = gd_ffn["indices"][b].tolist()
            entry["ffn_gate_probs"] = gd_ffn["probs"][b]

            routing_data.append(entry)

    for h in hooks:
        h.remove()

    return routing_data


def compute_routing_stats(routing_data, num_super_cats, num_proj_layers=2,
                          num_proj_experts=4, num_ffn_experts=8):
    """
    Aggregate routing data into statistics for visualization.

    Returns dict with:
        proj_heatmaps: list of 6 arrays, each (num_super_cats, num_proj_experts)
            — fraction of images from each super-cat routed to each expert
        ffn_heatmap: array (num_super_cats, num_ffn_experts)
        expert_entropy: dict of per-layer per-expert entropy values
        mutual_info: dict of per-layer normalized mutual information
    """
    import numpy as np
    from scipy.stats import entropy

    N_sup = num_super_cats

    # --- Projection MoE heatmaps ---
    proj_heatmaps = []
    for layer_i in range(num_proj_layers):
        heatmap = np.zeros((N_sup, num_proj_experts))
        for d in routing_data:
            sup = d["super_cat"]
            exp = d["proj_experts"][layer_i]
            heatmap[sup, exp] += 1

        # Normalize per super-category (row-wise)
        row_sums = heatmap.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        heatmap = heatmap / row_sums
        proj_heatmaps.append(heatmap)

    # --- FFN MoE heatmap (top-2, count both selected experts) ---
    ffn_heatmap = np.zeros((N_sup, num_ffn_experts))
    for d in routing_data:
        sup = d["super_cat"]
        for exp in d["ffn_experts"]:
            ffn_heatmap[sup, exp] += 1

    row_sums = ffn_heatmap.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    ffn_heatmap = ffn_heatmap / row_sums

    # --- Expert specialization entropy ---
    # For each expert: what's the entropy of its super-category distribution?
    # Low entropy = specialist, high entropy = generalist
    expert_entropy = {}

    for layer_i in range(num_proj_layers):
        expert_sup_counts = np.zeros((num_proj_experts, N_sup))
        for d in routing_data:
            exp = d["proj_experts"][layer_i]
            sup = d["super_cat"]
            expert_sup_counts[exp, sup] += 1

        layer_entropy = []
        for e in range(num_proj_experts):
            total = expert_sup_counts[e].sum()
            if total == 0:
                layer_entropy.append(0.0)
            else:
                dist = expert_sup_counts[e] / total
                layer_entropy.append(float(entropy(dist + 1e-10)))
        expert_entropy[f"proj_{layer_i}"] = layer_entropy

    # FFN entropy
    ffn_sup_counts = np.zeros((num_ffn_experts, N_sup))
    for d in routing_data:
        for exp in d["ffn_experts"]:
            sup = d["super_cat"]
            ffn_sup_counts[exp, sup] += 1

    ffn_ent = []
    for e in range(num_ffn_experts):
        total = ffn_sup_counts[e].sum()
        if total == 0:
            ffn_ent.append(0.0)
        else:
            dist = ffn_sup_counts[e] / total
            ffn_ent.append(float(entropy(dist + 1e-10)))
    expert_entropy["ffn"] = ffn_ent

    # --- Normalized Mutual Information ---
    mutual_info = {}
    for layer_i in range(num_proj_layers):
        experts = [d["proj_experts"][layer_i] for d in routing_data]
        supers = [d["super_cat"] for d in routing_data]
        mutual_info[f"proj_{layer_i}"] = _normalized_mutual_info(experts, supers)

    ffn_experts_flat = []
    ffn_supers_flat = []
    for d in routing_data:
        for exp in d["ffn_experts"]:
            ffn_experts_flat.append(exp)
            ffn_supers_flat.append(d["super_cat"])
    mutual_info["ffn"] = _normalized_mutual_info(ffn_experts_flat, ffn_supers_flat)

    return {
        "proj_heatmaps": proj_heatmaps,
        "ffn_heatmap": ffn_heatmap,
        "expert_entropy": expert_entropy,
        "mutual_info": mutual_info,
    }


def _normalized_mutual_info(labels_a, labels_b):
    """Compute normalized mutual information between two label lists."""
    from sklearn.metrics import normalized_mutual_info_score
    return float(normalized_mutual_info_score(labels_a, labels_b))
