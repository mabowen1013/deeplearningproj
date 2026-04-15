"""
Expert specialization visualizations for the report.

Generates:
  1. Expert Usage Heatmaps — expert x super-category routing frequency
  2. t-SNE of FFN gate logits — colored by super-category
  3. Expert Specialization Entropy — bar chart per layer
  4. NMI Summary — mutual information between routing and taxonomy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_expert_heatmaps(proj_heatmaps, ffn_heatmap, super_cat_names, save_dir,
                         proj_layer_names=None):
    """
    Plot expert usage heatmaps: rows = super-categories, cols = experts.

    One figure per MoE layer + one for FFN.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_proj = len(proj_heatmaps)
    if proj_layer_names is None:
        proj_layer_names = [f"Proj MoE {i}" for i in range(n_proj)]

    # Individual heatmaps
    for i, (heatmap, name) in enumerate(zip(proj_heatmaps, proj_layer_names)):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            heatmap, annot=True, fmt=".2f", cmap="YlOrRd",
            xticklabels=[f"E{e}" for e in range(heatmap.shape[1])],
            yticklabels=super_cat_names,
            ax=ax, vmin=0, vmax=1,
        )
        ax.set_title(f"Proj MoE: {name}")
        ax.set_xlabel("Expert")
        ax.set_ylabel("Super-category")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"heatmap_proj_{i}.png"), dpi=150)
        plt.close(fig)

    # FFN heatmap
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(
        ffn_heatmap, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=[f"E{e}" for e in range(ffn_heatmap.shape[1])],
        yticklabels=super_cat_names,
        ax=ax, vmin=0,
    )
    ax.set_title("MoE FFN Block (Top-2)")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Super-category")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "heatmap_ffn.png"), dpi=150)
    plt.close(fig)

    # Combined figure: all projection layers side by side
    if n_proj > 0:
        ncols = min(n_proj, 3)
        nrows = (n_proj + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                 squeeze=False)
        for i, (heatmap, name) in enumerate(zip(proj_heatmaps, proj_layer_names)):
            ax = axes[i // ncols, i % ncols]
            sns.heatmap(
                heatmap, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=[f"E{e}" for e in range(heatmap.shape[1])],
                yticklabels=super_cat_names if i % ncols == 0 else [],
                ax=ax, vmin=0, vmax=1, cbar=i % ncols == ncols - 1,
            )
            ax.set_title(name, fontsize=10)
            if i // ncols == nrows - 1:
                ax.set_xlabel("Expert")
        # Hide unused axes
        for j in range(n_proj, nrows * ncols):
            axes[j // ncols, j % ncols].set_visible(False)

        fig.suptitle("Expert Routing Patterns Across Projection MoE Layers", fontsize=13)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "heatmap_all_proj.png"), dpi=150)
        plt.close(fig)

    print(f"  Heatmaps saved to {save_dir}")


def plot_gate_tsne(routing_data, super_cat_names, save_dir):
    """
    t-SNE of FFN gate probabilities colored by super-category.

    If the gate learned taxonomy, clusters should correspond to super-categories.
    """
    from sklearn.manifold import TSNE

    os.makedirs(save_dir, exist_ok=True)

    # Collect FFN gate probs
    gate_probs = np.array([d["ffn_gate_probs"].numpy() for d in routing_data])
    super_cats = np.array([d["super_cat"] for d in routing_data])

    # t-SNE (adjust perplexity for small datasets)
    perplexity = min(30, len(gate_probs) // 4)
    if perplexity < 2:
        print("  WARNING: Too few samples for t-SNE, skipping")
        return

    # Skip if all rows are nearly identical (untrained model)
    row_std = np.std(gate_probs, axis=0).mean()  # variance across samples
    if row_std < 1e-6:
        print("  WARNING: Gate probs identical across samples, skipping t-SNE")
        return

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    embeddings = tsne.fit_transform(gate_probs)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(super_cat_names)))

    for idx, name in enumerate(super_cat_names):
        mask = super_cats == idx
        ax.scatter(
            embeddings[mask, 0], embeddings[mask, 1],
            c=[colors[idx]], label=name, s=8, alpha=0.6,
        )

    ax.legend(fontsize=9, markerscale=3)
    ax.set_title("t-SNE of FFN Gate Probabilities by Super-category")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "tsne_ffn_gates.png"), dpi=150)
    plt.close(fig)

    print(f"  t-SNE plot saved to {save_dir}")


def plot_expert_entropy(expert_entropy, save_dir, proj_layer_names=None):
    """
    Bar chart of per-expert entropy across layers.
    Low entropy = specialist, high entropy = generalist.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_proj = sum(1 for k in expert_entropy if k.startswith("proj_"))
    if proj_layer_names is None:
        proj_layer_names = [f"Proj{i}" for i in range(num_proj)]
    layer_names = list(proj_layer_names) + ["FFN"]
    layer_keys = [f"proj_{i}" for i in range(num_proj)] + ["ffn"]

    fig, ax = plt.subplots(figsize=(12, 4))

    x_offset = 0
    tick_positions = []
    tick_labels = []

    for layer_name, layer_key in zip(layer_names, layer_keys):
        entropies = expert_entropy[layer_key]
        n_experts = len(entropies)
        x = np.arange(n_experts) + x_offset

        bars = ax.bar(x, entropies, width=0.7, label=layer_name)

        tick_positions.extend(x.tolist())
        tick_labels.extend([f"{layer_name}\nE{e}" for e in range(n_experts)])

        x_offset += n_experts + 1  # gap between layers

    # Max entropy line (uniform distribution over super-categories)
    num_super = max(len(v) for v in expert_entropy.values())
    max_ent = np.log(num_super) if num_super > 0 else np.log(6)
    ax.axhline(y=max_ent, color='red', linestyle='--', alpha=0.5,
               label=f'Max entropy (uniform, {max_ent:.2f})')

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=7, rotation=0)
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Expert Specialization Entropy (lower = more specialized)")
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "expert_entropy.png"), dpi=150)
    plt.close(fig)

    print(f"  Entropy plot saved to {save_dir}")


def plot_nmi_summary(mutual_info, save_dir, proj_layer_names=None):
    """
    Bar chart of Normalized Mutual Information per layer.
    Higher NMI = stronger expert-taxonomy correspondence.
    """
    os.makedirs(save_dir, exist_ok=True)

    num_proj = sum(1 for k in mutual_info if k.startswith("proj_"))
    if proj_layer_names is None:
        proj_layer_names = [f"Proj{i}" for i in range(num_proj)]
    layer_names = list(proj_layer_names) + ["FFN"]
    layer_keys = [f"proj_{i}" for i in range(num_proj)] + ["ffn"]

    nmi_values = [mutual_info[k] for k in layer_keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(layer_names, nmi_values, color='steelblue')

    # Add value labels
    for bar, val in zip(bars, nmi_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Normalized Mutual Information")
    ax.set_title("Expert-Taxonomy Correspondence (NMI) per MoE Layer")
    ax.set_ylim(0, max(nmi_values) * 1.3 if max(nmi_values) > 0 else 0.1)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "nmi_summary.png"), dpi=150)
    plt.close(fig)

    print(f"  NMI plot saved to {save_dir}")


def generate_all_figures(routing_data, stats, super_cat_names, save_dir,
                         proj_layer_names=None):
    """Generate all analysis figures for the report."""
    print(f"\nGenerating analysis figures...")

    plot_expert_heatmaps(
        stats["proj_heatmaps"], stats["ffn_heatmap"],
        super_cat_names, save_dir,
        proj_layer_names=proj_layer_names,
    )

    plot_gate_tsne(routing_data, super_cat_names, save_dir)

    plot_expert_entropy(stats["expert_entropy"], save_dir,
                        proj_layer_names=proj_layer_names)

    plot_nmi_summary(stats["mutual_info"], save_dir,
                     proj_layer_names=proj_layer_names)

    print(f"\nAll figures saved to {save_dir}")
