#!/usr/bin/env python3
"""
Pre-generate dataset splits for all devices.

This script pre-computes the dataset indices for all devices to avoid
expensive on-the-fly Dirichlet sampling during training. The indices are
saved to a file that can be loaded quickly by each device.

Usage:
    python prepare_dataset_splits.py --num_devices 1000 --data_root /tmp/data
"""

import argparse
import os
import pickle
import time
from collections import defaultdict

import filelock
import numpy as np
from torchvision import datasets, transforms


def sample_dirichlet_indices(train_set, device_id, subset_size, dirichlet_alpha, seed=None):
    """
    Sample indices using Dirichlet distribution for non-IID data.

    Args:
        train_set: The training dataset
        device_id: Device ID for reproducibility
        subset_size: Number of samples per device
        dirichlet_alpha: Dirichlet concentration parameter
        seed: Random seed (default: based on device_id)

    Returns:
        List of indices for this device
    """
    num_classes = 10  # CIFAR10 has 10 classes

    # Generate device-specific seed for reproducibility
    if seed is None:
        seed = hash(str(device_id)) % (2**32)
    rng = np.random.RandomState(seed)

    # Memory-efficient: use dataset.targets directly
    if hasattr(train_set, "targets"):
        all_labels = np.array(train_set.targets)
    else:
        all_labels = np.array([train_set[i][1] for i in range(len(train_set))])

    # Create indices for each class
    class_indices = {k: np.where(all_labels == k)[0] for k in range(num_classes)}

    # Sample class proportions from Dirichlet distribution
    class_probs = rng.dirichlet([dirichlet_alpha] * num_classes)

    # Determine number of samples per class based on Dirichlet probabilities
    samples_per_class = (class_probs * subset_size).astype(int)

    # Adjust to ensure exactly subset_size samples
    diff = subset_size - samples_per_class.sum()
    if diff > 0:
        # Add remaining samples to classes with highest probability
        top_classes = np.argsort(class_probs)[-diff:]
        for cls in top_classes:
            samples_per_class[cls] += 1
    elif diff < 0:
        # Remove excess samples from classes with lowest probability
        bottom_classes = np.argsort(class_probs)[: abs(diff)]
        for cls in bottom_classes:
            samples_per_class[cls] = max(0, samples_per_class[cls] - 1)

    # Sample indices from each class
    selected_indices = []
    for cls in range(num_classes):
        n_samples = samples_per_class[cls]
        if n_samples > 0:
            cls_indices = class_indices[cls]
            if len(cls_indices) > 0:
                # Sample without replacement
                sampled = rng.choice(cls_indices, size=min(n_samples, len(cls_indices)), replace=False)
                selected_indices.extend(sampled.tolist())

    # Shuffle the selected indices
    rng.shuffle(selected_indices)

    return selected_indices[:subset_size]


def sample_iid_indices(train_set, device_id, subset_size, seed=None):
    """
    Sample indices using IID (random) distribution.

    Args:
        train_set: The training dataset
        device_id: Device ID for reproducibility
        subset_size: Number of samples per device
        seed: Random seed (default: based on device_id)

    Returns:
        List of indices for this device
    """
    # Generate device-specific seed for reproducibility
    if seed is None:
        seed = hash(str(device_id)) % (2**32)
    rng = np.random.RandomState(seed)

    # Random sampling
    indices = list(range(len(train_set)))
    rng.shuffle(indices)
    return indices[:subset_size]


def generate_dataset_splits(
    data_root: str,
    num_devices: int,
    subset_size: int,
    data_distribution: str = "non_iid_dirichlet",
    dirichlet_alpha: float = 0.5,
    output_file: str = None,
):
    """
    Generate and save dataset splits for all devices.

    Args:
        data_root: Root directory for CIFAR10 data
        num_devices: Number of devices to generate splits for
        subset_size: Number of samples per device
        data_distribution: Type of data distribution ('iid' or 'non_iid_dirichlet')
        dirichlet_alpha: Dirichlet concentration parameter (for non-IID)
        output_file: Output file path (default: auto-generated)
    """
    print("=" * 80)
    print("DATASET SPLIT GENERATION")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Data root:         {data_root}")
    print(f"  Number of devices: {num_devices}")
    print(f"  Samples per device: {subset_size}")
    print(f"  Distribution:      {data_distribution}")
    if data_distribution == "non_iid_dirichlet":
        print(f"  Dirichlet alpha:   {dirichlet_alpha}")
    print()

    # Load CIFAR10 dataset
    print("Loading CIFAR10 dataset...")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
    )

    # Add file lock to prevent multiple simultaneous downloads
    lock_file = os.path.join(data_root, "cifar10.lock")
    with filelock.FileLock(lock_file):
        train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)

    print(f"Dataset loaded: {len(train_set)} samples")
    print()

    # Generate splits for all devices
    print(f"Generating splits for {num_devices} devices...")
    start_time = time.time()

    device_splits = {}
    class_distributions = defaultdict(lambda: np.zeros(10))  # Track class distribution per device

    for device_id in range(num_devices):
        if (device_id + 1) % 100 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (device_id + 1)
            remaining = avg_time * (num_devices - device_id - 1)
            print(
                f"  Progress: {device_id + 1}/{num_devices} devices "
                f"({elapsed:.1f}s elapsed, ~{remaining:.1f}s remaining)"
            )

        if data_distribution == "non_iid_dirichlet":
            indices = sample_dirichlet_indices(train_set, device_id, subset_size, dirichlet_alpha)
        else:  # iid
            indices = sample_iid_indices(train_set, device_id, subset_size)

        device_splits[device_id] = indices

        # Track class distribution
        if hasattr(train_set, "targets"):
            for idx in indices:
                label = train_set.targets[idx]
                class_distributions[device_id][label] += 1

    total_time = time.time() - start_time
    print(f"\nGeneration complete in {total_time:.2f}s ({total_time/num_devices*1000:.2f}ms per device)")
    print()

    # Save to file
    if output_file is None:
        output_file = os.path.join(
            data_root, f"dataset_splits_{data_distribution}_n{num_devices}_s{subset_size}_a{dirichlet_alpha}.pkl"
        )

    print(f"Saving splits to: {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)

    split_data = {
        "device_splits": device_splits,
        "num_devices": num_devices,
        "subset_size": subset_size,
        "data_distribution": data_distribution,
        "dirichlet_alpha": dirichlet_alpha,
        "total_samples": len(train_set),
    }

    with open(output_file, "wb") as f:
        pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Splits saved successfully ({file_size_mb:.2f} MB)")
    print()

    # Print statistics
    print("=" * 80)
    print("DATASET SPLIT STATISTICS")
    print("=" * 80)

    # Calculate class distribution statistics
    all_distributions = np.array([class_distributions[i] for i in range(num_devices)])
    mean_dist = np.mean(all_distributions, axis=0)
    std_dist = np.std(all_distributions, axis=0)

    print(f"\n{'Class':<10} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-" * 70)
    for cls in range(10):
        print(
            f"{cls:<10} {mean_dist[cls]:<15.2f} {std_dist[cls]:<15.2f} "
            f"{np.min(all_distributions[:, cls]):<15.0f} {np.max(all_distributions[:, cls]):<15.0f}"
        )

    print()

    # Calculate entropy as a measure of non-IID-ness
    entropies = []
    for device_id in range(num_devices):
        probs = class_distributions[device_id] / subset_size
        probs = probs[probs > 0]  # Remove zeros to avoid log(0)
        entropy = -np.sum(probs * np.log(probs))
        entropies.append(entropy)

    print(f"Class Distribution Entropy (higher = more uniform):")
    print(f"  Mean:   {np.mean(entropies):.4f}")
    print(f"  Std:    {np.std(entropies):.4f}")
    print(f"  Min:    {np.min(entropies):.4f}")
    print(f"  Max:    {np.max(entropies):.4f}")
    print(f"  Max possible (uniform): {np.log(10):.4f}")
    print()

    print("=" * 80)
    print(f"Pre-generated splits ready to use!")
    print(f"File: {output_file}")
    print("=" * 80)

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Pre-generate dataset splits for federated learning")
    parser.add_argument(
        "--data_root", type=str, default="/tmp/data", help="Root directory for CIFAR10 data (default: /tmp/data)"
    )
    parser.add_argument(
        "--num_devices", type=int, default=1000, help="Number of devices to generate splits for (default: 1000)"
    )
    parser.add_argument("--subset_size", type=int, default=350, help="Number of samples per device (default: 350)")
    parser.add_argument(
        "--distribution",
        type=str,
        default="non_iid_dirichlet",
        choices=["iid", "non_iid_dirichlet"],
        help="Data distribution type (default: non_iid_dirichlet)",
    )
    parser.add_argument(
        "--dirichlet_alpha", type=float, default=0.5, help="Dirichlet concentration parameter (default: 0.5)"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output file path (default: auto-generated in data_root)"
    )

    args = parser.parse_args()

    output_file = generate_dataset_splits(
        data_root=args.data_root,
        num_devices=args.num_devices,
        subset_size=args.subset_size,
        data_distribution=args.distribution,
        dirichlet_alpha=args.dirichlet_alpha,
        output_file=args.output,
    )

    print(f"\nTo use these splits, pass the following to your task processor:")
    print(f"  dataset_splits_file='{output_file}'")


if __name__ == "__main__":
    main()
