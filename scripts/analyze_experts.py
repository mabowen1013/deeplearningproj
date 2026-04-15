"""
Expert specialization analysis entry point.

Runs val set through the trained MoE model, collects routing data,
computes statistics, and generates all report figures.

Usage:
    python scripts/analyze_experts.py --checkpoint checkpoints/moe/best.pt
"""

import os
import sys
import argparse

sys.path.insert(0, "D:/documents/duke/2026_spring/deep_learning/final_project")

import torch
from torch.utils.data import DataLoader

from src.model.efficientnet_moe import EfficientNetMoE, MOE_TARGETS
from src.data.dataset import INat2019Dataset
from src.data.transforms import get_val_transforms
from src.analysis.routing_analysis import collect_routing_data, compute_routing_stats
from src.analysis.taxonomy_viz import generate_all_figures


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="D:/documents/duke/2026_spring/deep_learning/final_project/data")
    parser.add_argument("--output-dir", type=str,
                        default="D:/documents/duke/2026_spring/deep_learning/final_project/results/analysis")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-classes-per-super", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load val data first to determine num_classes
    print("Loading val data...")
    val_dataset = INat2019Dataset(
        root=args.data_root, split="val", transform=get_val_transforms(),
        max_classes_per_super=args.max_classes_per_super,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    num_classes = val_dataset.num_classes

    # Load model
    print("Loading model...")
    model = EfficientNetMoE(num_classes=num_classes, pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded from epoch {ckpt.get('epoch', '?')}, "
          f"best_acc={ckpt.get('best_acc', '?')}")
    print(f"  {len(val_dataset)} images, {val_dataset.num_classes} species, "
          f"{val_dataset.num_super_categories} super-categories")

    # Collect routing data
    print("\nCollecting expert routing data...")
    routing_data = collect_routing_data(model, val_loader, device)
    print(f"  Collected {len(routing_data)} entries")

    # Save raw routing data
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(routing_data, f"{args.output_dir}/routing_data.pt")

    # Build layer names from MOE_TARGETS
    num_proj = len(MOE_TARGETS)
    proj_layer_names = [f"S{s}.B{b}" for s, b, *_ in MOE_TARGETS]

    # Compute statistics
    print("\nComputing routing statistics...")
    stats = compute_routing_stats(
        routing_data,
        num_super_cats=val_dataset.num_super_categories,
        num_proj_layers=num_proj,
    )

    # Print NMI summary
    print("\n--- Normalized Mutual Information ---")
    all_layer_names = proj_layer_names + ["FFN"]
    layer_keys = [f"proj_{i}" for i in range(num_proj)] + ["ffn"]
    for name, key in zip(all_layer_names, layer_keys):
        nmi = stats["mutual_info"][key]
        print(f"  {name:8s}: NMI = {nmi:.4f}")

    # Generate figures
    generate_all_figures(
        routing_data, stats,
        val_dataset.super_categories,
        args.output_dir,
        proj_layer_names=proj_layer_names,
    )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
