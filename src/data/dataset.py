"""
iNaturalist 2019 dataset.

Reads directly from the extracted directory structure and handles
train/val split internally (stratified: 3 images per species for val,
matching the original iNat 2019 competition setup).

Expected layout:
    data_root/
      train_val2019/
        Amphibians/
          153/
            img1.jpg ...
          154/
            ...
        Birds/
          ...
"""

import os
import random
from collections import Counter, defaultdict

import torch
from PIL import Image
from torch.utils.data import Dataset


class INat2019Dataset(Dataset):
    """
    iNaturalist 2019 dataset with built-in train/val split.

    Returns: image, species_label, super_category_label
    """

    def __init__(self, root: str, split: str = "train", transform=None,
                 val_per_class: int = 3, seed: int = 42,
                 max_classes_per_super: int = None,
                 max_images_per_class: int = None):
        """
        Args:
            root: path containing 'train_val2019/' directory
            split: 'train' or 'val'
            transform: image transforms
            val_per_class: number of images per species held out for val
            seed: random seed for reproducible split
            max_classes_per_super: if set, subsample species per super-category
            max_images_per_class: if set, cap training images per species
        """
        self.root = root
        self.split = split
        self.transform = transform

        data_dir = os.path.join(root, "train_val2019")
        assert os.path.isdir(data_dir), f"Data dir not found: {data_dir}"

        # Discover super-categories and species
        self.super_categories = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        self.num_super_categories = len(self.super_categories)
        self.super_cat_to_idx = {name: i for i, name in enumerate(self.super_categories)}

        # Build full index: list of (image_path, species_id, super_cat_idx)
        # species_id is a global 0-based index across all species
        rng = random.Random(seed)

        # First pass: discover all species per super-category
        super_to_species_dirs = {}
        for super_name in self.super_categories:
            super_dir = os.path.join(data_dir, super_name)
            species_dirs = sorted([
                d for d in os.listdir(super_dir)
                if os.path.isdir(os.path.join(super_dir, d))
            ])
            # Subsample species if requested
            if max_classes_per_super and len(species_dirs) > max_classes_per_super:
                species_dirs = sorted(rng.sample(species_dirs, max_classes_per_super))
            super_to_species_dirs[super_name] = species_dirs

        # Second pass: build samples with contiguous species IDs
        all_samples = []
        species_to_idx = {}
        global_species_id = 0

        for super_name in self.super_categories:
            super_idx = self.super_cat_to_idx[super_name]
            super_dir = os.path.join(data_dir, super_name)

            for species_dir_name in super_to_species_dirs[super_name]:
                species_path = os.path.join(super_dir, species_dir_name)

                species_key = f"{super_name}/{species_dir_name}"
                if species_key not in species_to_idx:
                    species_to_idx[species_key] = global_species_id
                    global_species_id += 1

                species_id = species_to_idx[species_key]

                images = sorted([
                    f for f in os.listdir(species_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])

                for img_name in images:
                    img_path = os.path.join(species_path, img_name)
                    all_samples.append((img_path, species_id, super_idx))

        self.num_classes = global_species_id
        self.species_to_super = {}
        for (_, sp_id, sup_idx) in all_samples:
            self.species_to_super[sp_id] = sup_idx

        # Stratified train/val split: hold out val_per_class images per species
        samples_by_species = defaultdict(list)
        for sample in all_samples:
            samples_by_species[sample[1]].append(sample)

        train_samples = []
        val_samples = []
        for species_id in sorted(samples_by_species.keys()):
            imgs = samples_by_species[species_id]
            rng.shuffle(imgs)
            val_samples.extend(imgs[:val_per_class])
            train_imgs = imgs[val_per_class:]
            # Cap training images per class if requested
            if max_images_per_class and len(train_imgs) > max_images_per_class:
                train_imgs = train_imgs[:max_images_per_class]
            train_samples.extend(train_imgs)

        if split == "train":
            self.samples = train_samples
        elif split == "val":
            self.samples = val_samples
        else:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")

        # Pre-compute labels for efficient access
        self.species_labels = [s[1] for s in self.samples]
        self.class_counts = Counter(self.species_labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, species_label, super_label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, species_label, super_label

    def get_sample_weights(self):
        """Per-sample weights for WeightedRandomSampler (1/class_count)."""
        weights = [1.0 / self.class_counts[label] for label in self.species_labels]
        return torch.tensor(weights, dtype=torch.float64)

    def get_super_category_name(self, super_idx: int) -> str:
        return self.super_categories[super_idx]
