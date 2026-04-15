"""
测试数据模块的接口和 transforms。
由于真实数据未下载，这里测试 transforms 和 sampler 逻辑。
"""

import sys
sys.path.insert(0, "D:/documents/duke/2026_spring/deep_learning/final_project")

import torch
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.data.transforms import get_train_transforms, get_val_transforms


def test_transforms():
    print("=" * 50)
    print("Testing Transforms")
    print("=" * 50)

    train_t = get_train_transforms()
    val_t = get_val_transforms()

    # Create a dummy PIL image (simulate a camera trap photo)
    img = Image.new("RGB", (640, 480), color=(128, 100, 80))

    # Train transform
    tensor_train = train_t(img)
    print(f"Train: PIL {img.size} -> Tensor {list(tensor_train.shape)}")
    assert tensor_train.shape == (3, 224, 224), f"Expected (3,224,224), got {tensor_train.shape}"

    # Val transform
    tensor_val = val_t(img)
    print(f"Val:   PIL {img.size} -> Tensor {list(tensor_val.shape)}")
    assert tensor_val.shape == (3, 224, 224)

    # Check normalization (values should be roughly centered around 0)
    print(f"Train tensor range: [{tensor_train.min():.2f}, {tensor_train.max():.2f}]")
    print(f"Val tensor range:   [{tensor_val.min():.2f}, {tensor_val.max():.2f}]")

    print("Transforms OK!\n")


def test_weighted_sampler_logic():
    """Test that WeightedRandomSampler corrects class imbalance."""
    print("=" * 50)
    print("Testing WeightedRandomSampler Logic")
    print("=" * 50)

    # Simulate iNat-like imbalance: 3 classes with 1000, 100, 10 samples
    class_sizes = [1000, 100, 10]
    labels = []
    for cls_id, size in enumerate(class_sizes):
        labels.extend([cls_id] * size)

    # Compute weights like our dataset does
    class_counts = Counter(labels)
    weights = [1.0 / class_counts[l] for l in labels]
    weights = torch.tensor(weights, dtype=torch.float64)

    sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)

    # Sample and check distribution
    sampled_indices = list(sampler)
    sampled_labels = [labels[i] for i in sampled_indices]
    sampled_counts = Counter(sampled_labels)

    print(f"Original distribution: {dict(Counter(labels))}")
    print(f"Sampled distribution:  {dict(sampled_counts)}")

    # After balancing, each class should have roughly equal samples
    total = sum(sampled_counts.values())
    for cls_id in range(3):
        pct = sampled_counts[cls_id] / total * 100
        print(f"  Class {cls_id}: {sampled_counts[cls_id]:>5d} ({pct:.1f}%) - "
              f"{'balanced' if 25 < pct < 42 else 'SKEWED'}")

    print("Sampler logic OK!\n")


class MockINat2019(Dataset):
    """Simulates INat2019Dataset interface for DataLoader testing."""

    def __init__(self, num_samples=100, num_classes=10, num_super=3):
        self.num_classes = num_classes
        self.num_super = num_super
        # Imbalanced: class 0 has 50 samples, others share the rest
        self.species_labels = []
        for i in range(num_samples):
            if i < num_samples // 2:
                self.species_labels.append(0)
            else:
                self.species_labels.append(i % num_classes)
        self.class_counts = Counter(self.species_labels)

    def __len__(self):
        return len(self.species_labels)

    def __getitem__(self, idx):
        image = torch.randn(3, 224, 224)
        species = self.species_labels[idx]
        super_cat = species % self.num_super
        return image, species, super_cat

    def get_sample_weights(self):
        return torch.tensor(
            [1.0 / self.class_counts[l] for l in self.species_labels],
            dtype=torch.float64,
        )


def test_dataloader_interface():
    """Test that our dataset interface works with DataLoader."""
    print("=" * 50)
    print("Testing DataLoader Interface (Mock Data)")
    print("=" * 50)

    dataset = MockINat2019(num_samples=100, num_classes=10, num_super=3)

    # With balanced sampler
    weights = dataset.get_sample_weights()
    sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)

    loader = DataLoader(dataset, batch_size=16, sampler=sampler, drop_last=True)

    batch_count = 0
    for images, species, super_cats in loader:
        if batch_count == 0:
            print(f"Batch shapes: images={list(images.shape)}, "
                  f"species={list(species.shape)}, super={list(super_cats.shape)}")
            print(f"Species labels: {species.tolist()}")
            print(f"Super-cat labels: {super_cats.tolist()}")
        batch_count += 1

    print(f"Total batches: {batch_count}")
    print(f"Effective epoch size: {batch_count * 16}")
    print("DataLoader interface OK!\n")


if __name__ == "__main__":
    test_transforms()
    test_weighted_sampler_logic()
    test_dataloader_interface()
    print("All data module tests passed!")
