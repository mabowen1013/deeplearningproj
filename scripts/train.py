"""
Training entry point for EfficientNet-MoE on iNaturalist 2019.

Usage:
    python scripts/train.py
    python scripts/train.py --resume checkpoints/latest.pt
    python scripts/train.py --baseline  # train vanilla EfficientNet-B0
"""

import sys
import argparse

sys.path.insert(0, "D:/documents/duke/2026_spring/deep_learning/final_project")

import torch
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.model.efficientnet_moe import EfficientNetMoE
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.sampler import build_dataloaders
from src.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train EfficientNet-MoE")
    parser.add_argument("--data-root", type=str,
                        default="D:/documents/duke/2026_spring/deep_learning/final_project/data")
    parser.add_argument("--output-dir", type=str,
                        default="D:/documents/duke/2026_spring/deep_learning/final_project/checkpoints/moe")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--baseline", action="store_true",
                        help="Train vanilla EfficientNet-B0 (no MoE) as baseline")

    # Training hyperparams
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr-backbone", type=float, default=1e-4)
    parser.add_argument("--lr-new", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 is fastest on Windows)")

    # Data subsampling
    parser.add_argument("--max-classes-per-super", type=int, default=None,
                        help="Max species per super-category (None=all)")
    parser.add_argument("--max-images-per-class", type=int, default=None,
                        help="Max training images per species (None=all)")

    # Loss
    parser.add_argument("--alpha", type=float, default=0.01,
                        help="Load-balancing loss weight")
    parser.add_argument("--beta", type=float, default=0.001,
                        help="Router z-loss weight")
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    return parser.parse_args()


def build_baseline_model(num_classes=1010):
    """Build vanilla EfficientNet-B0 for baseline comparison."""
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(1280, num_classes)
    return model


class BaselineTrainerAdapter:
    """
    Wraps vanilla EfficientNet-B0 to match the MoE trainer interface.
    The model returns (logits, dummy_aux_losses) so the same Trainer works.
    """

    def __init__(self, model):
        self._model = model
        # Expose the same attributes as EfficientNetMoE
        self.moe_wrappers = torch.nn.ModuleList()
        self.moe_ffn = None
        self.classifier = model.classifier

    def __getattr__(self, name):
        if name in ('_model', 'moe_wrappers', 'moe_ffn', 'classifier'):
            return super().__getattribute__(name)
        return getattr(self._model, name)

    def __call__(self, x):
        logits = self._model(x)
        dummy_aux = {"load_balance_loss": torch.tensor(0.0, device=x.device),
                     "z_loss": torch.tensor(0.0, device=x.device)}
        return logits, dummy_aux

    def to(self, device):
        self._model = self._model.to(device)
        return self

    def train(self, mode=True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self

    def parameters(self):
        return self._model.parameters()

    def named_parameters(self):
        return self._model.named_parameters()

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, state_dict):
        return self._model.load_state_dict(state_dict)


def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Data (load first to determine num_classes)
    print(f"\nLoading data from {args.data_root}...")
    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        data_root=args.data_root,
        train_transform=get_train_transforms(),
        val_transform=get_val_transforms(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_classes_per_super=args.max_classes_per_super,
        max_images_per_class=args.max_images_per_class,
    )
    num_classes = train_ds.num_classes
    print(f"  Train: {len(train_ds)} images, {num_classes} species, "
          f"{train_ds.num_super_categories} super-categories")
    print(f"  Val:   {len(val_ds)} images")
    print(f"  Batch size: {args.batch_size} x {args.grad_accum} accum = "
          f"{args.batch_size * args.grad_accum} effective")

    # Model
    if args.baseline:
        print("\nBuilding baseline EfficientNet-B0...")
        model = BaselineTrainerAdapter(build_baseline_model(num_classes=num_classes))
        args.output_dir = args.output_dir.replace("/moe", "/baseline")
    else:
        print("\nBuilding EfficientNet-MoE...")
        model = EfficientNetMoE(num_classes=num_classes, pretrained=True)
        stats = model.get_param_stats()
        print(f"  Total params:  {stats['total']:,}")
        print(f"  Active params: {stats['active']:,}")

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=args.output_dir,
        lr_backbone=args.lr_backbone,
        lr_new=args.lr_new,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        grad_accum_steps=args.grad_accum,
        alpha=args.alpha,
        beta=args.beta,
        label_smoothing=args.label_smoothing,
    )

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
