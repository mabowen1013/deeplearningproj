"""
Class-balanced sampling for long-tailed iNaturalist 2019.
"""

import platform

from torch.utils.data import DataLoader, WeightedRandomSampler

from .dataset import INat2019Dataset

# Windows multiprocessing spawn overhead makes num_workers>0 slower
_DEFAULT_WORKERS = 0 if platform.system() == "Windows" else 4


def build_dataloaders(
    data_root: str,
    train_transform,
    val_transform,
    batch_size: int = 32,
    num_workers: int = _DEFAULT_WORKERS,
    pin_memory: bool = True,
    max_classes_per_super: int = None,
    max_images_per_class: int = None,
):
    """
    Build train and val DataLoaders with class-balanced sampling for train.

    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    train_dataset = INat2019Dataset(
        root=data_root, split="train", transform=train_transform,
        max_classes_per_super=max_classes_per_super,
        max_images_per_class=max_images_per_class,
    )
    val_dataset = INat2019Dataset(
        root=data_root, split="val", transform=val_transform,
        max_classes_per_super=max_classes_per_super,
    )

    # Class-balanced sampler for imbalanced training data
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_dataset, val_dataset
