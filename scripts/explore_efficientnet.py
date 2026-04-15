"""
探索 EfficientNet-B0 的架构：打印每层名称、形状、参数量。
帮助我们确定哪些层适合替换为 MoE。
"""

import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def main():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    # 1. 总参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{'='*70}")
    print(f"EfficientNet-B0 Total Parameters: {total_params:,}")
    print(f"20% budget cap: {int(total_params * 1.2):,}")
    print(f"{'='*70}\n")

    # 2. 顶层模块概览
    print("Top-level modules:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name:20s} -> {params:>10,} params")
    print()

    # 3. features 内部每个 block 的详细结构
    print(f"{'='*70}")
    print("Detailed features (MBConv blocks):")
    print(f"{'='*70}\n")

    for i, block in enumerate(model.features):
        block_params = sum(p.numel() for p in block.parameters())
        print(f"features[{i}]: {block.__class__.__name__} -> {block_params:,} params")

        # 打印子模块
        for name, sub in block.named_modules():
            if name == "":
                continue
            sub_params = sum(p.numel() for p in sub.parameters(recurse=False))
            if sub_params > 0:
                # 获取层的形状信息
                shape_info = ""
                if hasattr(sub, 'weight') and sub.weight is not None:
                    shape_info = f"  weight: {list(sub.weight.shape)}"
                if hasattr(sub, 'in_features'):
                    shape_info = f"  ({sub.in_features} -> {sub.out_features})"
                if hasattr(sub, 'in_channels'):
                    shape_info = f"  ({sub.in_channels} -> {sub.out_channels}, k={sub.kernel_size})"

                print(f"    {name:40s} {sub_params:>8,} params{shape_info}")
        print()

    # 4. classifier
    print(f"{'='*70}")
    print("Classifier:")
    for name, sub in model.classifier.named_modules():
        if name == "":
            continue
        sub_params = sum(p.numel() for p in sub.parameters(recurse=False))
        if sub_params > 0:
            shape_info = ""
            if hasattr(sub, 'in_features'):
                shape_info = f"  ({sub.in_features} -> {sub.out_features})"
            print(f"    {name:40s} {sub_params:>8,} params{shape_info}")

    # 5. 按 stage 汇总（EfficientNet-B0 的 stage 划分）
    print(f"\n{'='*70}")
    print("Summary by stage:")
    print(f"{'='*70}")

    # features[0] = stem conv
    # features[1] = stage 1 (MBConv1, 1 block)
    # features[2] = stage 2 (MBConv6, 2 blocks)
    # features[3] = stage 3 (MBConv6, 2 blocks)
    # features[4] = stage 4 (MBConv6, 3 blocks)
    # features[5] = stage 5 (MBConv6, 3 blocks)
    # features[6] = stage 6 (MBConv6, 4 blocks)
    # features[7] = stage 7 (MBConv6, 1 block)
    # features[8] = final conv

    stage_names = [
        "Stem Conv", "Stage 1 (MBConv1)", "Stage 2 (MBConv6)",
        "Stage 3 (MBConv6)", "Stage 4 (MBConv6)", "Stage 5 (MBConv6)",
        "Stage 6 (MBConv6)", "Stage 7 (MBConv6)", "Final Conv 1x1"
    ]

    for i, name in enumerate(stage_names):
        block = model.features[i]
        params = sum(p.numel() for p in block.parameters())
        pct = params / total_params * 100
        # 如果是 Sequential，统计子 block 数
        if hasattr(block, '__len__'):
            n_blocks = len(block)
            print(f"  features[{i}] {name:25s} -> {params:>10,} params ({pct:5.1f}%)  [{n_blocks} blocks]")
        else:
            print(f"  features[{i}] {name:25s} -> {params:>10,} params ({pct:5.1f}%)")

    cls_params = sum(p.numel() for p in model.classifier.parameters())
    print(f"  {'classifier':33s} -> {cls_params:>10,} params ({cls_params/total_params*100:5.1f}%)")

    # 6. 模拟一次前向传播，打印每个 stage 的输出形状
    print(f"\n{'='*70}")
    print("Feature map shapes (input: 1x3x224x224):")
    print(f"{'='*70}")

    x = torch.randn(1, 3, 224, 224)
    for i, block in enumerate(model.features):
        x = block(x)
        print(f"  features[{i}] output: {list(x.shape)}")


if __name__ == "__main__":
    main()
