"""
Evaluation script for EfficientNet-MoE.

Reports:
  - Top-1 accuracy (overall and per-super-category)
  - Parameter counts (total vs active)
  - FLOPs comparison with baseline
  - Expert utilization statistics

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/moe/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/baseline/best.pt --baseline
"""

import sys
import argparse
from collections import defaultdict, Counter

sys.path.insert(0, "D:/documents/duke/2026_spring/deep_learning/final_project")

import torch
from torch.amp import autocast

from src.model.efficientnet_moe import EfficientNetMoE
from src.model.param_counter import (
    count_baseline_params_and_flops,
    count_moe_params_and_flops,
    format_params,
    format_flops,
)
from src.data.dataset import INat2019Dataset
from src.data.transforms import get_val_transforms
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str,
                        default="D:/documents/duke/2026_spring/deep_learning/final_project/data")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--baseline", action="store_true",
                        help="Evaluate vanilla EfficientNet-B0")
    parser.add_argument("--max-classes-per-super", type=int, default=None)
    return parser.parse_args()


def load_model(checkpoint_path, num_classes, baseline=False, device="cuda"):
    """Load model from checkpoint."""
    if baseline:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(1280, num_classes)
    else:
        model = EfficientNetMoE(num_classes=num_classes, pretrained=False)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
    print(f"  Best acc: {ckpt.get('best_acc', '?')}")
    return model


@torch.no_grad()
def evaluate_accuracy(model, val_loader, dataset, device, is_baseline=False):
    """Compute top-1 accuracy overall and per super-category."""
    model.eval()

    all_preds = []
    all_species = []
    all_supers = []

    # Expert routing stats (MoE only)
    routing_counts = defaultdict(lambda: Counter())  # layer -> Counter of expert_ids

    for images, species, super_cats in val_loader:
        images = images.to(device, non_blocking=True)

        with autocast("cuda"):
            if is_baseline:
                logits = model(images)
            else:
                logits, _ = model(images)

        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_species.extend(species.tolist())
        all_supers.extend(super_cats.tolist())

        # Collect expert routing (MoE only)
        if not is_baseline and hasattr(model, 'moe_wrappers'):
            for i, wrapper in enumerate(model.moe_wrappers):
                if hasattr(wrapper, 'aux_losses') and wrapper.aux_losses:
                    pass  # routing info is in the gate, collected below

    # Overall accuracy
    correct = sum(p == t for p, t in zip(all_preds, all_species))
    total = len(all_preds)
    overall_acc = correct / total * 100

    # Per-super-category accuracy
    super_correct = defaultdict(int)
    super_total = defaultdict(int)
    for pred, true_sp, true_sup in zip(all_preds, all_species, all_supers):
        super_total[true_sup] += 1
        if pred == true_sp:
            super_correct[true_sup] += 1

    return {
        "overall_acc": overall_acc,
        "correct": correct,
        "total": total,
        "per_super": {
            sup_idx: super_correct[sup_idx] / super_total[sup_idx] * 100
            for sup_idx in sorted(super_total.keys())
        },
        "per_super_total": dict(super_total),
    }


@torch.no_grad()
def collect_expert_routing(model, val_loader, device):
    """
    Run val set and collect expert routing decisions for each MoE layer.

    Returns:
        routing_data: list of dicts, one per sample:
            {species, super_cat, proj_experts: [6 ints], ffn_experts: [2 ints]}
    """
    model.eval()
    routing_data = []

    # Register hooks on gate modules to capture expert indices
    gate_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            weights, indices, _ = output
            gate_outputs[name] = indices.cpu()
        return hook

    hooks = []
    for i, wrapper in enumerate(model.moe_wrappers):
        h = wrapper.moe.gate.register_forward_hook(make_hook(f"proj_{i}"))
        hooks.append(h)

    h = model.moe_ffn.gate.register_forward_hook(make_hook("ffn"))
    hooks.append(h)

    for images, species, super_cats in val_loader:
        images = images.to(device, non_blocking=True)
        gate_outputs.clear()

        with autocast("cuda"):
            model(images)

        batch_size = images.size(0)
        for b in range(batch_size):
            entry = {
                "species": species[b].item(),
                "super_cat": super_cats[b].item(),
            }
            # Projection MoE experts (top-1 each)
            entry["proj_experts"] = [
                gate_outputs[f"proj_{i}"][b].item()
                for i in range(len(model.moe_wrappers))
            ]
            # FFN MoE experts (top-2)
            entry["ffn_experts"] = gate_outputs["ffn"][b].tolist()

            routing_data.append(entry)

    for h in hooks:
        h.remove()

    return routing_data


def print_report(acc_results, param_stats, dataset, baseline_stats=None):
    """Print formatted evaluation report."""
    print("\n" + "=" * 65)
    print("EVALUATION REPORT")
    print("=" * 65)

    # Accuracy
    print(f"\n--- Top-1 Accuracy ---")
    print(f"  Overall: {acc_results['overall_acc']:.2f}% "
          f"({acc_results['correct']}/{acc_results['total']})")

    print(f"\n  Per super-category:")
    for sup_idx in sorted(acc_results['per_super'].keys()):
        name = dataset.get_super_category_name(sup_idx)
        acc = acc_results['per_super'][sup_idx]
        n = acc_results['per_super_total'][sup_idx]
        print(f"    {name:15s}: {acc:5.1f}%  ({n} samples)")

    # Parameters
    print(f"\n--- Parameters ---")
    print(f"  Total:    {format_params(param_stats['total_params']):>10s}")
    print(f"  Active:   {format_params(param_stats['active_params']):>10s}")
    if 'inactive_params' in param_stats:
        print(f"  Inactive: {format_params(param_stats['inactive_params']):>10s}")

    # Comparison with baseline
    if baseline_stats:
        print(f"\n--- Comparison with Baseline ---")
        base_p = baseline_stats['total_params']
        our_total = param_stats['total_params']
        our_active = param_stats['active_params']
        overhead = (our_total / base_p - 1) * 100
        savings = (1 - our_active / base_p) * 100

        print(f"  Baseline total params:  {format_params(base_p)}")
        print(f"  MoE total params:       {format_params(our_total)} ({overhead:+.1f}%)")
        print(f"  MoE active params:      {format_params(our_active)} ({savings:+.1f}% savings)")
        print(f"  Budget check:           {'PASS' if overhead <= 20 else 'FAIL'} (limit: +20%)")
        print(f"  Active check:           {'PASS' if our_active < base_p else 'FAIL'} (must be < baseline)")

        if 'flops' in baseline_stats and 'flops' in param_stats:
            print(f"\n  Baseline FLOPs: {format_flops(baseline_stats['flops'])}")
            print(f"  MoE FLOPs:      {format_flops(param_stats['flops'])}")

    print("\n" + "=" * 65)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load val data first to determine num_classes
    val_dataset = INat2019Dataset(
        root=args.data_root, split="val", transform=get_val_transforms(),
        max_classes_per_super=args.max_classes_per_super,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    num_classes = val_dataset.num_classes
    print(f"Val set: {len(val_dataset)} images, {num_classes} species")

    # Load model
    model = load_model(args.checkpoint, num_classes=num_classes,
                       baseline=args.baseline, device=device)

    # Evaluate accuracy
    print("\nEvaluating accuracy...")
    acc_results = evaluate_accuracy(
        model, val_loader, val_dataset, device, is_baseline=args.baseline,
    )

    # Parameter and FLOP stats
    baseline_stats = count_baseline_params_and_flops()
    if args.baseline:
        param_stats = baseline_stats
    else:
        param_stats = count_moe_params_and_flops(model)

    # Print report
    print_report(
        acc_results, param_stats, val_dataset,
        baseline_stats=None if args.baseline else baseline_stats,
    )

    # Collect and save routing data (MoE only)
    if not args.baseline:
        print("\nCollecting expert routing data...")
        routing_data = collect_expert_routing(model, val_loader, device)

        save_path = args.checkpoint.replace(".pt", "_routing.pt")
        torch.save(routing_data, save_path)
        print(f"Saved routing data ({len(routing_data)} samples) to {save_path}")

        # Quick routing summary
        num_proj = len(model.moe_wrappers)
        num_proj_experts = model.moe_wrappers[0].moe.num_experts if num_proj > 0 else 0
        num_ffn_experts = model.moe_ffn.num_experts

        print(f"\n--- Expert Utilization (val set) ---")
        for layer_i in range(num_proj):
            counts = Counter(d["proj_experts"][layer_i] for d in routing_data)
            total = sum(counts.values())
            dist = " | ".join(f"E{e}:{counts.get(e,0)/total*100:.0f}%"
                              for e in range(num_proj_experts))
            print(f"  Proj[{layer_i}]: {dist}")

        ffn_counts = Counter()
        for d in routing_data:
            for e in d["ffn_experts"]:
                ffn_counts[e] += 1
        total_ffn = sum(ffn_counts.values())
        ffn_dist = " | ".join(f"E{e}:{ffn_counts.get(e,0)/total_ffn*100:.0f}%"
                               for e in range(num_ffn_experts))
        print(f"  FFN:     {ffn_dist}")


if __name__ == "__main__":
    main()
