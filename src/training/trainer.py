"""
Training loop for EfficientNet-MoE.

Features:
  - Two-phase training: freeze backbone first, then full fine-tune
  - Mixed precision (AMP) for 8GB VRAM
  - Gradient accumulation for effective batch size
  - Expert utilization tracking
  - Best model checkpointing
"""

import os
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from ..model.efficientnet_moe import EfficientNetMoE
from .losses import MoELoss
from .scheduler import get_cosine_with_warmup


class Trainer:

    def __init__(
        self,
        model: EfficientNetMoE,
        train_loader,
        val_loader,
        device: torch.device,
        output_dir: str,
        # Training hyperparams
        lr_backbone: float = 1e-4,
        lr_new: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        freeze_epochs: int = 5,
        grad_accum_steps: int = 2,
        # Loss weights
        alpha: float = 0.01,
        beta: float = 0.001,
        label_smoothing: float = 0.1,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = output_dir
        self.total_epochs = total_epochs
        self.freeze_epochs = freeze_epochs
        self.grad_accum_steps = grad_accum_steps

        os.makedirs(output_dir, exist_ok=True)

        # Loss
        self.criterion = MoELoss(
            label_smoothing=label_smoothing, alpha=alpha, beta=beta,
        )

        # Optimizer with parameter groups
        self.optimizer = self._build_optimizer(lr_backbone, lr_new, weight_decay)

        # Scheduler
        self.scheduler = get_cosine_with_warmup(
            self.optimizer, warmup_epochs, total_epochs,
        )

        # Mixed precision
        self.scaler = GradScaler("cuda")

        # Tracking
        self.best_acc = 0.0
        self.start_epoch = 1
        self.history = defaultdict(list)

    def _build_optimizer(self, lr_backbone, lr_new, weight_decay):
        """Separate param groups: backbone (low lr) vs new MoE+classifier (high lr)."""
        backbone_params = []
        new_params = []

        # MoE wrappers and FFN block are new (may be empty/None for baseline)
        new_module_ids = set()
        if hasattr(self.model, 'moe_wrappers'):
            for wrapper in self.model.moe_wrappers:
                new_module_ids.update(id(p) for p in wrapper.parameters())
        if hasattr(self.model, 'moe_ffn') and self.model.moe_ffn is not None:
            for p in self.model.moe_ffn.parameters():
                new_module_ids.add(id(p))
        for p in self.model.classifier.parameters():
            new_module_ids.add(id(p))

        for param in self.model.parameters():
            if id(param) in new_module_ids:
                new_params.append(param)
            else:
                backbone_params.append(param)

        return torch.optim.AdamW([
            {"params": backbone_params, "lr": lr_backbone},
            {"params": new_params, "lr": lr_new},
        ], weight_decay=weight_decay)

    def _freeze_backbone(self):
        """Freeze all backbone params, only train MoE + classifier."""
        for name, param in self.model.named_parameters():
            is_moe = "moe_wrappers" in name or "moe_ffn" in name
            is_cls = "classifier" in name
            param.requires_grad = is_moe or is_cls

    def _unfreeze_all(self):
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True

    def train(self):
        """Main training loop."""
        print(f"Training for {self.total_epochs} epochs (starting from {self.start_epoch})")
        print(f"  Phase 1 (epochs 1-{self.freeze_epochs}): backbone frozen")
        print(f"  Phase 2 (epochs {self.freeze_epochs+1}-{self.total_epochs}): full fine-tune")
        print(f"  Grad accumulation: {self.grad_accum_steps} steps")
        print(f"  Output: {self.output_dir}")
        print()

        for epoch in range(self.start_epoch, self.total_epochs + 1):
            # Phase switching
            if epoch <= self.freeze_epochs:
                self._freeze_backbone()
                if epoch == self.start_epoch:
                    print("Phase 1: Backbone FROZEN")
            elif epoch == self.start_epoch or epoch == self.freeze_epochs + 1:
                self._unfreeze_all()
                print("Phase 2: Full fine-tuning UNLOCKED")

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate(epoch)

            # Step scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Log
            self._log_epoch(epoch, train_metrics, val_metrics, current_lr)

            # Save checkpoint
            is_best = val_metrics["acc"] > self.best_acc
            if is_best:
                self.best_acc = val_metrics["acc"]
            self._save_checkpoint(epoch, val_metrics, is_best)

        print(f"\nTraining complete. Best val accuracy: {self.best_acc:.2f}%")

    def _train_epoch(self, epoch):
        self.model.train()
        metrics = defaultdict(float)
        total_correct = 0
        total_samples = 0

        self.optimizer.zero_grad()

        for step, (images, species, _super) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            species = species.to(self.device, non_blocking=True)

            # Forward with mixed precision
            with autocast("cuda"):
                logits, aux_losses = self.model(images)
                loss, loss_dict = self.criterion(logits, species, aux_losses)
                loss = loss / self.grad_accum_steps  # scale for accumulation

            # Backward
            self.scaler.scale(loss).backward()

            # Optimizer step every grad_accum_steps
            if (step + 1) % self.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Track metrics
            batch_size = images.size(0)
            total_samples += batch_size
            total_correct += (logits.argmax(dim=1) == species).sum().item()
            for k, v in loss_dict.items():
                metrics[k] += v * batch_size

            # Print progress
            if (step + 1) % 100 == 0:
                avg_loss = metrics["total"] / total_samples
                acc = total_correct / total_samples * 100
                print(f"  Epoch {epoch} [{step+1}/{len(self.train_loader)}] "
                      f"loss={avg_loss:.4f} acc={acc:.1f}%", end="\r")

        # Average metrics
        for k in metrics:
            metrics[k] /= total_samples
        metrics["acc"] = total_correct / total_samples * 100

        return dict(metrics)

    @torch.no_grad()
    def _validate(self, epoch):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        metrics = defaultdict(float)

        for images, species, _super in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            species = species.to(self.device, non_blocking=True)

            with autocast("cuda"):
                logits, aux_losses = self.model(images)
                _, loss_dict = self.criterion(logits, species, aux_losses)

            batch_size = images.size(0)
            total_samples += batch_size
            total_correct += (logits.argmax(dim=1) == species).sum().item()
            for k, v in loss_dict.items():
                metrics[k] += v * batch_size

        for k in metrics:
            metrics[k] /= total_samples
        metrics["acc"] = total_correct / total_samples * 100

        return dict(metrics)

    def _log_epoch(self, epoch, train_m, val_m, lr):
        self.history["epoch"].append(epoch)
        self.history["lr"].append(lr)
        for prefix, m in [("train", train_m), ("val", val_m)]:
            for k, v in m.items():
                self.history[f"{prefix}_{k}"].append(v)

        star = " *" if val_m["acc"] >= self.best_acc else ""
        print(f"Epoch {epoch:3d} | "
              f"train loss={train_m['total']:.4f} acc={train_m['acc']:.1f}% | "
              f"val loss={val_m['total']:.4f} acc={val_m['acc']:.1f}%{star} | "
              f"bal={val_m['balance']:.3f} lr={lr:.2e}")

    def _save_checkpoint(self, epoch, val_metrics, is_best):
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "best_acc": self.best_acc,
            "val_metrics": val_metrics,
            "history": dict(self.history),
        }

        # Always save latest
        torch.save(state, os.path.join(self.output_dir, "latest.pt"))

        # Save best
        if is_best:
            torch.save(state, os.path.join(self.output_dir, "best.pt"))
            print(f"  -> New best model saved (acc={val_metrics['acc']:.2f}%)")

    def load_checkpoint(self, path: str):
        """Resume training from checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])
        self.best_acc = ckpt["best_acc"]
        self.history = defaultdict(list, ckpt["history"])
        self.start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (best_acc={self.best_acc:.2f}%)")
