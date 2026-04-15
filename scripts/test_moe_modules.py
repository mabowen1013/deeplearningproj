"""
测试 MoE 核心模块：NoisyTopKGate 和 FactoredMoEProjection。
验证形状正确、损失正常、参数计数合理。
"""

import sys
sys.path.insert(0, "D:/documents/duke/2026_spring/deep_learning/final_project")

import torch
from src.model.gating import NoisyTopKGate
from src.model.moe_layer import FactoredMoEProjection


def test_gate():
    print("=" * 50)
    print("Testing NoisyTopKGate")
    print("=" * 50)

    gate = NoisyTopKGate(input_dim=672, num_experts=4, top_k=1)
    x = torch.randn(8, 672)  # batch of 8, 672-dim features

    # Training mode
    gate.train()
    weights, indices, aux = gate(x)
    print(f"[Train] weights shape: {weights.shape}")  # (8, 1)
    print(f"[Train] indices shape: {indices.shape}")  # (8, 1)
    print(f"[Train] load_balance_loss: {aux['load_balance_loss']:.4f}")
    print(f"[Train] z_loss: {aux['z_loss']:.4f}")
    print(f"[Train] weights sum per sample: {weights.sum(dim=-1)}")  # should be all 1.0

    # Eval mode (no noise)
    gate.eval()
    weights, indices, aux = gate(x)
    print(f"\n[Eval]  weights shape: {weights.shape}")
    print(f"[Eval]  indices: {indices.squeeze()}")
    print(f"[Eval]  weights: {weights.squeeze()}")  # should be all 1.0 for top-1

    # Top-2 test
    gate2 = NoisyTopKGate(input_dim=192, num_experts=8, top_k=2)
    gate2.eval()
    x2 = torch.randn(4, 192)
    weights2, indices2, _ = gate2(x2)
    print(f"\n[Top-2] weights shape: {weights2.shape}")  # (4, 2)
    print(f"[Top-2] indices: {indices2}")
    print(f"[Top-2] weights sum: {weights2.sum(dim=-1)}")  # should be all 1.0

    print("\nGate tests passed!\n")


def test_moe_projection():
    print("=" * 50)
    print("Testing FactoredMoEProjection")
    print("=" * 50)

    # Simulate replacing features[5][1].block.3.0: Conv2d(672 -> 112)
    moe = FactoredMoEProjection(
        in_channels=672,
        out_channels=112,
        num_experts=4,
        top_k=1,
        bottleneck_rank=32,
    )

    x = torch.randn(8, 672, 14, 14)  # Stage 5 feature map

    # Training
    moe.train()
    out, aux = moe(x)
    print(f"Input shape:  {list(x.shape)}")
    print(f"Output shape: {list(out.shape)}")  # should be (8, 112, 14, 14)
    print(f"Load balance loss: {aux['load_balance_loss']:.4f}")
    print(f"Z-loss: {aux['z_loss']:.4f}")

    # Parameter count
    total_params = sum(p.numel() for p in moe.parameters())
    active_params = moe.count_active_params()
    original_params = 672 * 112  # original Conv2d(672, 112, 1x1) without bias
    print(f"\nOriginal Conv2d params:    {original_params:,}")
    print(f"MoE total params:          {total_params:,}")
    print(f"MoE active params (top-1): {active_params:,}")
    print(f"Total overhead:            +{total_params - original_params:,} ({(total_params/original_params - 1)*100:.1f}%)")
    print(f"Active savings:            -{original_params - active_params:,} ({(1 - active_params/original_params)*100:.1f}%)")

    # Gradient check
    loss = out.sum() + aux['load_balance_loss'] + aux['z_loss']
    loss.backward()
    print(f"\nGradient check:")
    for name, p in moe.named_parameters():
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        print(f"  {name:40s} grad: {'OK' if has_grad else 'NONE'}")

    print("\nMoE projection tests passed!\n")


def test_stage6_moe():
    print("=" * 50)
    print("Testing Stage 6 MoE (1152 -> 192)")
    print("=" * 50)

    moe = FactoredMoEProjection(
        in_channels=1152,
        out_channels=192,
        num_experts=4,
        top_k=1,
        bottleneck_rank=48,
    )

    x = torch.randn(8, 1152, 7, 7)
    moe.train()
    out, aux = moe(x)
    print(f"Input:  {list(x.shape)}")
    print(f"Output: {list(out.shape)}")

    total = sum(p.numel() for p in moe.parameters())
    active = moe.count_active_params()
    original = 1152 * 192
    print(f"Original: {original:,} | MoE total: {total:,} | MoE active: {active:,}")
    print(f"Overhead: +{(total/original-1)*100:.1f}% | Savings: -{(1-active/original)*100:.1f}%")

    print("\nStage 6 tests passed!\n")


if __name__ == "__main__":
    test_gate()
    test_moe_projection()
    test_stage6_moe()
    print("All tests passed!")
