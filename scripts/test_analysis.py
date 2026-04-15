"""
测试分析模块：用未训练模型 + 真实 val 子集，验证完整 pipeline。
"""

import sys
sys.path.insert(0, "D:/documents/duke/2026_spring/deep_learning/final_project")

import torch
from torch.utils.data import DataLoader, Subset

from src.model.efficientnet_moe import EfficientNetMoE
from src.data.dataset import INat2019Dataset
from src.data.transforms import get_val_transforms
from src.analysis.routing_analysis import collect_routing_data, compute_routing_stats
from src.analysis.taxonomy_viz import generate_all_figures


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Untrained model
    model = EfficientNetMoE(num_classes=1010, pretrained=False).to(device)

    # Real val data — use 200 samples for quick test
    val_dataset = INat2019Dataset(
        root="D:/documents/duke/2026_spring/deep_learning/final_project/data",
        split="val",
        transform=get_val_transforms(),
    )
    subset = Subset(val_dataset, range(min(200, len(val_dataset))))
    loader = DataLoader(subset, batch_size=32, num_workers=0)
    print(f"Using {len(subset)} val images")

    # Step 1: Collect routing data
    print("\n1. Collecting routing data...")
    routing_data = collect_routing_data(model, loader, device)
    print(f"   Got {len(routing_data)} entries")

    sample = routing_data[0]
    print(f"   Sample: species={sample['species']}, super={sample['super_cat']}")
    print(f"     proj_experts={sample['proj_experts']}")
    print(f"     ffn_experts={sample['ffn_experts']}")
    print(f"     ffn_gate_probs shape={sample['ffn_gate_probs'].shape}")

    # Step 2: Compute stats
    print("\n2. Computing routing stats...")
    stats = compute_routing_stats(routing_data, num_super_cats=val_dataset.num_super_categories)

    print(f"   proj_heatmaps: {len(stats['proj_heatmaps'])} layers")
    print(f"   ffn_heatmap shape: {stats['ffn_heatmap'].shape}")

    print("\n   NMI per layer:")
    for key, nmi in stats["mutual_info"].items():
        print(f"     {key}: {nmi:.4f}")

    print("\n   Expert entropy (FFN):")
    for e, ent in enumerate(stats["expert_entropy"]["ffn"]):
        print(f"     Expert {e}: {ent:.3f}")

    # Step 3: Generate figures
    print("\n3. Generating figures...")
    save_dir = "D:/documents/duke/2026_spring/deep_learning/final_project/results/test_analysis"
    generate_all_figures(routing_data, stats, val_dataset.super_categories, save_dir)

    # Verify files exist
    import os
    generated = os.listdir(save_dir)
    print(f"\n   Generated {len(generated)} files:")
    for f in sorted(generated):
        size_kb = os.path.getsize(os.path.join(save_dir, f)) / 1024
        print(f"     {f} ({size_kb:.0f} KB)")

    # Cleanup
    import shutil
    shutil.rmtree(save_dir)
    print("\n   Test files cleaned up")

    print("\n" + "=" * 50)
    print("All analysis tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
