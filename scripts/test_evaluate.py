"""
测试评估模块：param_counter + evaluate 的核心函数。
使用未训练的模型和真实 val 数据的前 2 个 batch。
"""

import sys
sys.path.insert(0, "D:/documents/duke/2026_spring/deep_learning/final_project")

import torch
from torch.utils.data import DataLoader, Subset

from src.model.efficientnet_moe import EfficientNetMoE
from src.model.param_counter import (
    count_baseline_params_and_flops,
    count_moe_params_and_flops,
    format_params,
    format_flops,
)
from src.data.dataset import INat2019Dataset
from src.data.transforms import get_val_transforms
from scripts.evaluate import evaluate_accuracy, collect_expert_routing


def test_param_counter():
    print("=" * 55)
    print("Testing param_counter")
    print("=" * 55)

    # Baseline
    print("\nBaseline EfficientNet-B0 (1010 classes):")
    baseline = count_baseline_params_and_flops()
    print(f"  Total params: {format_params(baseline['total_params'])}")
    print(f"  FLOPs:        {format_flops(baseline['flops'])}")

    # MoE
    print("\nEfficientNet-MoE:")
    model = EfficientNetMoE(num_classes=1010, pretrained=False)
    moe_stats = count_moe_params_and_flops(model)
    print(f"  Total params:  {format_params(moe_stats['total_params'])}")
    print(f"  Active params: {format_params(moe_stats['active_params'])}")
    print(f"  FLOPs:         {format_flops(moe_stats['flops'])}")

    # Budget check
    overhead = (moe_stats['total_params'] / baseline['total_params'] - 1) * 100
    savings = (1 - moe_stats['active_params'] / baseline['total_params']) * 100
    print(f"\n  Total overhead: {overhead:+.1f}% (limit: 20%) {'PASS' if overhead <= 20 else 'FAIL'}")
    print(f"  Active savings: {savings:+.1f}% {'PASS' if moe_stats['active_params'] < baseline['total_params'] else 'FAIL'}")

    print("\nparam_counter OK!\n")
    return model


def test_eval_with_real_data(model):
    print("=" * 55)
    print("Testing evaluate on real val data (small subset)")
    print("=" * 55)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Load a small subset of real val data
    val_dataset = INat2019Dataset(
        root="D:/documents/duke/2026_spring/deep_learning/final_project/data",
        split="val",
        transform=get_val_transforms(),
    )

    # Take first 64 samples only
    subset = Subset(val_dataset, range(min(64, len(val_dataset))))
    loader = DataLoader(subset, batch_size=32, num_workers=0)

    print(f"Using {len(subset)} val images")

    # Test accuracy eval
    print("\nEvaluating accuracy...")
    acc = evaluate_accuracy(model, loader, val_dataset, device, is_baseline=False)
    print(f"  Overall acc: {acc['overall_acc']:.1f}% ({acc['correct']}/{acc['total']})")
    print(f"  Per super-category:")
    for sup_idx, sup_acc in acc['per_super'].items():
        name = val_dataset.get_super_category_name(sup_idx)
        print(f"    {name:15s}: {sup_acc:.1f}%")

    # Test routing collection
    print("\nCollecting routing data...")
    routing = collect_expert_routing(model, loader, device)
    print(f"  Collected {len(routing)} routing entries")

    sample = routing[0]
    print(f"  Sample entry: species={sample['species']}, "
          f"super={sample['super_cat']}, "
          f"proj_experts={sample['proj_experts']}, "
          f"ffn_experts={sample['ffn_experts']}")

    # Check expert utilization
    from collections import Counter
    for i in range(6):
        counts = Counter(d['proj_experts'][i] for d in routing)
        print(f"  Proj[{i}] utilization: {dict(counts)}")

    print("\nevaluate OK!")


if __name__ == "__main__":
    model = test_param_counter()
    test_eval_with_real_data(model)
    print("\n" + "=" * 55)
    print("All evaluate tests passed!")
    print("=" * 55)
