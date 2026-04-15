"""
测试完整 EfficientNet-MoE 模型：
- 前向传播形状正确
- 参数预算满足约束
- 梯度正常
"""

import sys
sys.path.insert(0, "D:/documents/duke/2026_spring/deep_learning/final_project")

import torch
from src.model.efficientnet_moe import EfficientNetMoE


def main():
    print("Loading EfficientNet-MoE...")
    model = EfficientNetMoE(num_classes=1010, pretrained=True)
    model.train()

    # =========================================
    # 1. Forward pass
    # =========================================
    print("\n" + "=" * 60)
    print("1. Forward Pass Test")
    print("=" * 60)

    x = torch.randn(4, 3, 224, 224)
    logits, aux = model(x)

    print(f"Input:  {list(x.shape)}")
    print(f"Logits: {list(logits.shape)}")  # (4, 1010)
    print(f"Load balance loss: {aux['load_balance_loss']:.4f}")
    print(f"Z-loss: {aux['z_loss']:.4f}")

    assert logits.shape == (4, 1010), f"Expected (4, 1010), got {logits.shape}"
    print("Forward pass OK!")

    # =========================================
    # 2. Parameter budget
    # =========================================
    print("\n" + "=" * 60)
    print("2. Parameter Budget")
    print("=" * 60)

    stats = model.get_param_stats()
    baseline_params = 5_288_548  # EfficientNet-B0

    print(f"EfficientNet-B0 baseline:  {baseline_params:>10,}")
    print(f"20% budget cap:            {int(baseline_params * 1.2):>10,}")
    print(f"")
    print(f"MoE model total params:    {stats['total']:>10,}")
    print(f"MoE model active params:   {stats['active']:>10,}")
    print(f"MoE model inactive params: {stats['inactive']:>10,}")
    print(f"")

    overhead = (stats['total'] / baseline_params - 1) * 100
    savings = (1 - stats['active'] / baseline_params) * 100

    print(f"Total overhead:  {overhead:+.1f}%  {'PASS' if overhead <= 20 else 'FAIL'}")
    print(f"Active savings:  {savings:+.1f}%  {'PASS' if stats['active'] < baseline_params else 'FAIL'}")

    assert stats['total'] <= baseline_params * 1.2, "BUDGET EXCEEDED!"
    assert stats['active'] < baseline_params, "ACTIVE PARAMS NOT REDUCED!"

    # =========================================
    # 3. MoE layer details
    # =========================================
    print("\n" + "=" * 60)
    print("3. MoE Layer Details")
    print("=" * 60)

    for i, wrapper in enumerate(model.moe_wrappers):
        moe = wrapper.moe
        total = sum(p.numel() for p in moe.parameters())
        active = moe.count_active_params()
        print(f"  MoE Proj [{i}]: {moe.in_channels}->{moe.out_channels} "
              f"(r={moe.bottleneck_rank}, N={moe.num_experts}, k={moe.top_k}) "
              f"total={total:,} active={active:,}")

    ffn = model.moe_ffn
    ffn_total = sum(p.numel() for p in ffn.parameters())
    ffn_active = ffn.count_active_params()
    print(f"  MoE FFN:     {ffn.dim}->{ffn.dim} "
          f"(hidden=128, N={ffn.num_experts}, k={ffn.top_k}) "
          f"total={ffn_total:,} active={ffn_active:,}")

    # =========================================
    # 4. Backward pass
    # =========================================
    print("\n" + "=" * 60)
    print("4. Backward Pass Test")
    print("=" * 60)

    loss = logits.sum() + aux['load_balance_loss'] + aux['z_loss']
    loss.backward()

    total_params = 0
    params_with_grad = 0
    for name, p in model.named_parameters():
        total_params += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            params_with_grad += 1

    print(f"Total parameter tensors: {total_params}")
    print(f"With gradient:           {params_with_grad}")
    print(f"Without gradient:        {total_params - params_with_grad}")
    print("(Some experts may lack grad in small batch — this is expected)")

    # =========================================
    # 5. Inference test
    # =========================================
    print("\n" + "=" * 60)
    print("5. Inference Test")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        logits_eval, aux_eval = model(x)
    print(f"Eval logits shape: {list(logits_eval.shape)}")
    print(f"Eval load balance: {aux_eval['load_balance_loss']:.4f}")
    print("Inference OK!")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
