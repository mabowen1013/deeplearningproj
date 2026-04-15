"""
下载 iNaturalist 2019 数据集。

使用流式解压：边下载边解压，不在磁盘上存储完整 .tar.gz。
磁盘峰值只需解压后的 ~75GB，而非传统方式的 ~148GB。

用法：
    python scripts/download_data.py
"""

import os
import sys
import tarfile
import shutil
import time
from collections import Counter
from urllib.request import urlopen, Request

DATA_ROOT = "D:/documents/duke/2026_spring/deep_learning/final_project/data"

DATASET_URL = "https://ml-inat-competition-datasets.s3.amazonaws.com/2019/train_val2019.tar.gz"
ARCHIVE_SIZE_GB = 73.1


def check_disk_space():
    total, used, free = shutil.disk_usage("D:/")
    free_gb = free / (1024**3)
    print(f"D: drive free space: {free_gb:.1f} GB")
    print(f"Estimated need:      ~75 GB (stream extraction, no archive stored)")
    if free_gb < 80:
        print("WARNING: Less than 80GB free. May not have enough space.")
        response = input("Continue? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(0)
    return free_gb


def stream_download_and_extract(url: str, extract_dir: str):
    """
    Download tar.gz from URL and extract on-the-fly.
    Never writes the full archive to disk.
    """
    print(f"Streaming download from:\n  {url}")
    print(f"Extracting to:\n  {extract_dir}")
    print(f"Archive size: ~{ARCHIVE_SIZE_GB} GB")
    print(f"This will take a while...\n")

    os.makedirs(extract_dir, exist_ok=True)

    req = Request(url)
    response = urlopen(req, timeout=60)

    # Track progress
    total_bytes = int(response.headers.get('Content-Length', 0))
    downloaded = 0
    start_time = time.time()
    last_print = 0
    file_count = 0

    # Use tarfile in streaming mode (r|gz = streaming gzip)
    with tarfile.open(fileobj=response, mode='r|gz') as tar:
        for member in tar:
            # Extract each file
            tar.extract(member, path=extract_dir)
            file_count += 1

            # Progress (approximate, based on tarfile position)
            elapsed = time.time() - start_time
            if elapsed - last_print > 30:  # print every 30 seconds
                last_print = elapsed
                elapsed_min = elapsed / 60
                print(f"  [{elapsed_min:.0f} min] Extracted {file_count:,} files...")

    elapsed = time.time() - start_time
    print(f"\nDone! Extracted {file_count:,} files in {elapsed/60:.1f} minutes")


def reorganize_for_torchvision(extract_dir: str):
    """
    The tar.gz extracts to a nested directory. Reorganize so torchvision
    can find the data at DATA_ROOT/2019_train/ and DATA_ROOT/2019_val/.

    Expected tar structure: train_val2019/ containing train/ and val/ dirs
    with super-category subdirectories.
    """
    # Check what got extracted
    extracted_items = os.listdir(extract_dir)
    print(f"\nExtracted top-level contents: {extracted_items}")

    # Look for the train_val2019 directory or similar
    for item in extracted_items:
        full_path = os.path.join(extract_dir, item)
        if os.path.isdir(full_path):
            sub_items = os.listdir(full_path)
            print(f"  {item}/: {sub_items[:10]}{'...' if len(sub_items) > 10 else ''}")


def print_stats_from_dir(data_root: str, split: str):
    """Print dataset statistics using torchvision."""
    from torchvision.datasets import INaturalist

    version = f"2019_{split}"
    version_dir = os.path.join(data_root, version)

    if not os.path.isdir(version_dir):
        print(f"  {version_dir} not found, skipping stats")
        return

    dataset = INaturalist(data_root, version, target_type=["full", "super"])

    print(f"\n--- {split} split ---")
    print(f"  Total images: {len(dataset)}")

    super_counts = Counter()
    species_counts = Counter()
    for cat_id, _ in dataset.index:
        species_counts[cat_id] += 1
        super_idx = dataset.categories_map[cat_id]["super"]
        super_counts[super_idx] += 1

    print(f"  Species: {len(species_counts)}")
    print(f"  Super-categories: {len(super_counts)}")

    # Super-category breakdown
    super_names = sorted([
        d for d in os.listdir(version_dir)
        if os.path.isdir(os.path.join(version_dir, d))
    ])
    print(f"\n  Super-category distribution:")
    for idx, name in enumerate(super_names):
        count = super_counts.get(idx, 0)
        print(f"    {name:20s}: {count:>6d} images")

    counts = list(species_counts.values())
    print(f"\n  Per-species: min={min(counts)}, max={max(counts)}, "
          f"mean={sum(counts)/len(counts):.1f}, "
          f"median={sorted(counts)[len(counts)//2]}")


def main():
    print("=" * 60)
    print("iNaturalist 2019 Dataset — Streaming Download")
    print("=" * 60)

    # Check if already exists
    train_dir = os.path.join(DATA_ROOT, "2019_train")
    val_dir = os.path.join(DATA_ROOT, "2019_val")

    if os.path.isdir(train_dir) and os.path.isdir(val_dir):
        print(f"\nDataset already exists at {DATA_ROOT}")
        print_stats_from_dir(DATA_ROOT, "train")
        print_stats_from_dir(DATA_ROOT, "val")
        return

    check_disk_space()

    # Stream download and extract
    stream_download_and_extract(DATASET_URL, DATA_ROOT)

    # Check structure and reorganize if needed
    reorganize_for_torchvision(DATA_ROOT)

    # Final disk space
    _, _, free = shutil.disk_usage("D:/")
    print(f"\nDisk space remaining: {free/(1024**3):.1f} GB")

    # Print stats if the directories are in the right place
    if os.path.isdir(train_dir):
        print_stats_from_dir(DATA_ROOT, "train")
    if os.path.isdir(val_dir):
        print_stats_from_dir(DATA_ROOT, "val")
    else:
        print("\nNOTE: You may need to rename/move the extracted directories")
        print(f"so that {train_dir} and {val_dir} exist.")
        print("Check the extracted structure above and adjust accordingly.")

    print("\nDone!")


if __name__ == "__main__":
    main()
