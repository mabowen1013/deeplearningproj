"""测试 MoEFFNBlock：形状、参数计数、梯度。"""

import sys
sys.path.insert(0, "D:/documents/duke/2026_spring/deep_learning/final_project")

import torch
from src.model.moe_ffn import MoEFFNBlock


def test_moe_ffn():
    print("=" * 50)
    print("Testing MoEFFNBlock (dim=192, hidden=128, N=8, k=2)")
    print("=" * 50)

    block = MoEFFNBlock(dim=192, hidden_dim=128, num_experts=8, top_k=2)

    # Simulate Stage 6 output: (B, 192, 7, 7)
    x = torch.randn(8, 192, 7, 7)

    # Training
    block.train()
    out, aux = block(x)
    print(f"Input shape:  {list(x.shape)}")
    print(f"Output shape: {list(out.shape)}")
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"

    # Residual: output should be close to input (since weights are randomly initialized)
    diff = (out - x).abs().mean().item()
    print(f"Mean |output - input|: {diff:.4f}  (residual sanity check)")

    print(f"\nLoad balance loss: {aux['load_balance_loss']:.4f}")
    print(f"Z-loss: {aux['z_loss']:.4f}")

    # Parameter counts
    total_params = sum(p.numel() for p in block.parameters())
    active_params = block.count_active_params()
    print(f"\nTotal params:  {total_params:,}")
    print(f"Active params: {active_params:,}  (top-2 of 8 experts)")

    # Breakdown
    gate_params = sum(p.numel() for p in block.gate.parameters())
    norm_params = sum(p.numel() for p in block.norm.parameters())
    expert_params = total_params - gate_params - norm_params
    print(f"\n  Gate params:    {gate_params:,}")
    print(f"  Norm params:    {norm_params:,}")
    print(f"  Expert params:  {expert_params:,}  (8 experts total)")
    print(f"  Per expert:     {expert_params // 8:,}")

    # Gradient check
    loss = out.sum() + aux['load_balance_loss'] + aux['z_loss']
    loss.backward()

    print(f"\nGradient check:")
    no_grad = []
    for name, p in block.named_parameters():
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        status = "OK" if has_grad else "NONE"
        if not has_grad:
            no_grad.append(name)
        print(f"  {name:40s} grad: {status}")

    if no_grad:
        print(f"\n  Note: {len(no_grad)} params without grad (w_noise is expected)")

    # Eval mode
    block.eval()
    with torch.no_grad():
        out_eval, aux_eval = block(x)
    print(f"\n[Eval] Output shape: {list(out_eval.shape)}")
    print(f"[Eval] Load balance loss: {aux_eval['load_balance_loss']:.4f}")

    print("\nMoEFFNBlock tests passed!")


if __name__ == "__main__":
    test_moe_ffn()
