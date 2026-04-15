"""
用 mock 数据快速测试训练模块：losses、scheduler、trainer 前向/反向。
"""

import sys
sys.path.insert(0, "D:/documents/duke/2026_spring/deep_learning/final_project")

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.model.efficientnet_moe import EfficientNetMoE
from src.training.losses import MoELoss
from src.training.scheduler import get_cosine_with_warmup
from src.training.trainer import Trainer


def test_loss():
    print("=" * 50)
    print("Testing MoELoss")
    print("=" * 50)

    criterion = MoELoss(num_classes=1010, label_smoothing=0.1, alpha=0.01, beta=0.001)

    logits = torch.randn(4, 1010)
    targets = torch.randint(0, 1010, (4,))
    aux = {"load_balance_loss": torch.tensor(1.05), "z_loss": torch.tensor(2.3)}

    total, loss_dict = criterion(logits, targets, aux)
    print(f"Total: {total.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    # Verify total = ce + alpha*balance + beta*z
    expected = loss_dict["ce"] + 0.01 * loss_dict["balance"] + 0.001 * loss_dict["z"]
    assert abs(total.item() - expected) < 1e-4, f"Loss mismatch: {total.item()} vs {expected}"
    print("Loss OK!\n")


def test_scheduler():
    print("=" * 50)
    print("Testing Cosine Scheduler with Warmup")
    print("=" * 50)

    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_with_warmup(optimizer, warmup_epochs=5, total_epochs=50)

    lrs = []
    for epoch in range(50):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()

    print(f"Epoch  0: lr={lrs[0]:.6f}  (should be ~0)")
    print(f"Epoch  2: lr={lrs[2]:.6f}  (warmup)")
    print(f"Epoch  5: lr={lrs[5]:.6f}  (peak)")
    print(f"Epoch 25: lr={lrs[25]:.6f}  (mid decay)")
    print(f"Epoch 49: lr={lrs[49]:.6f}  (near 0)")

    assert lrs[0] < lrs[5], "Warmup should increase LR"
    assert lrs[5] > lrs[25] > lrs[49], "Cosine should decay"
    print("Scheduler OK!\n")


def test_trainer_one_step():
    """Run 1 training step with tiny mock data to verify the full pipeline."""
    print("=" * 50)
    print("Testing Trainer (1 step, mock data)")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tiny model
    model = EfficientNetMoE(num_classes=10, pretrained=False)

    # Mock data: 8 samples
    images = torch.randn(8, 3, 224, 224)
    species = torch.randint(0, 10, (8,))
    super_cats = torch.randint(0, 3, (8,))

    train_ds = TensorDataset(images, species, super_cats)
    val_ds = TensorDataset(images, species, super_cats)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir="D:/documents/duke/2026_spring/deep_learning/final_project/checkpoints/test",
        total_epochs=2,
        freeze_epochs=1,
        warmup_epochs=1,
        grad_accum_steps=1,
    )

    # Run training
    trainer.train()

    # Check that checkpoints were saved
    import os
    ckpt_dir = "D:/documents/duke/2026_spring/deep_learning/final_project/checkpoints/test"
    assert os.path.exists(os.path.join(ckpt_dir, "latest.pt")), "latest.pt not saved"
    assert os.path.exists(os.path.join(ckpt_dir, "best.pt")), "best.pt not saved"

    # Cleanup test checkpoints
    import shutil
    shutil.rmtree(ckpt_dir)

    print("\nTrainer test OK!")


if __name__ == "__main__":
    test_loss()
    test_scheduler()
    test_trainer_one_step()
    print("\n" + "=" * 50)
    print("All training module tests passed!")
    print("=" * 50)
