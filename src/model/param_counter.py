"""
Parameter and FLOP counting utilities.

Reports total params, active params (per inference), and FLOPs
for both baseline EfficientNet-B0 and our MoE variant.
"""

import torch
import torch.nn as nn
from ..model.efficientnet_moe import EfficientNetMoE


def count_conv_flops(module: nn.Conv2d, input_h: int, input_w: int) -> int:
    """FLOPs for a single Conv2d: 2 * C_in * C_out * K^2 * H_out * W_out / groups."""
    out_h = input_h // module.stride[0]
    out_w = input_w // module.stride[1]
    k_h, k_w = module.kernel_size
    flops = 2 * module.in_channels * module.out_channels * k_h * k_w * out_h * out_w
    flops //= module.groups
    return flops


def count_linear_flops(module: nn.Linear) -> int:
    """FLOPs for a Linear layer: 2 * in * out."""
    return 2 * module.in_features * module.out_features


def count_baseline_params_and_flops():
    """Count params and FLOPs for vanilla EfficientNet-B0 (1010 classes)."""
    from torchvision.models import efficientnet_b0
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(1280, 1010)

    total_params = sum(p.numel() for p in model.parameters())

    # Use a forward hook to measure FLOPs
    x = torch.randn(1, 3, 224, 224)
    flops = _estimate_flops_with_hooks(model, x)

    return {
        "total_params": total_params,
        "active_params": total_params,  # all params active in dense model
        "flops": flops,
    }


def count_moe_params_and_flops(model: EfficientNetMoE):
    """Count params and FLOPs for our MoE model."""
    stats = model.get_param_stats()

    # Estimate active FLOPs
    x = torch.randn(1, 3, 224, 224)
    device = next(model.parameters()).device
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        flops = _estimate_flops_with_hooks(model, x)

    return {
        "total_params": stats["total"],
        "active_params": stats["active"],
        "inactive_params": stats["inactive"],
        "flops": flops,
    }


def _estimate_flops_with_hooks(model, x):
    """Estimate FLOPs by hooking into Conv2d and Linear layers."""
    flops_list = []

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            def hook_conv(mod, inp, out):
                _, _, h, w = inp[0].shape
                f = count_conv_flops(mod, h, w)
                flops_list.append(f)
            hooks.append(module.register_forward_hook(hook_conv))
        elif isinstance(module, nn.Linear):
            def hook_linear(mod, inp, out):
                flops_list.append(count_linear_flops(mod))
            hooks.append(module.register_forward_hook(hook_linear))

    model.eval()
    with torch.no_grad():
        try:
            out = model(x)
        except Exception:
            # For models returning tuple (logits, aux)
            pass

    for h in hooks:
        h.remove()

    return sum(flops_list)


def format_params(n: int) -> str:
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def format_flops(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.2f} GFLOPs"
    elif n >= 1e6:
        return f"{n/1e6:.1f} MFLOPs"
    return f"{n} FLOPs"
