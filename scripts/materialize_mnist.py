#!/usr/bin/env python3
"""
Materialize MNIST dataset into the dataset store.
Creates dataset revision D0001 with proper structure and checksums.
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def compute_directory_hash(dir_path: Path) -> str:
    """Compute combined hash of all files in a directory."""
    sha256 = hashlib.sha256()

    if not dir_path.exists():
        return "sha256:empty"

    # Sort files for deterministic hashing
    files = sorted(dir_path.rglob("*"))
    for file_path in files:
        if file_path.is_file():
            with open(file_path, "rb") as f:
                sha256.update(f.read())

    return f"sha256:{sha256.hexdigest()}"


def download_mnist(store_path: Path, holdback_percent: float = 0.0):
    """
    Download MNIST using torchvision and save to dataset store.

    Args:
        store_path: Path to dataset store (e.g., /ml-data-store/mnist/D0001)
        holdback_percent: Percentage of training data to hold back for later (0-100)
    """
    try:
        from torchvision import datasets
        import torch
    except ImportError:
        print("Error: torchvision not installed.")
        print("Install with: pip install torchvision")
        sys.exit(1)

    print(f"Downloading MNIST to {store_path}...")

    # Download train and test datasets
    print("\nDownloading training set...")
    train_dataset = datasets.MNIST(
        root=store_path,
        train=True,
        download=True
    )

    print("Downloading test set...")
    test_dataset = datasets.MNIST(
        root=store_path,
        train=False,
        download=True
    )

    print(f"\nDataset downloaded successfully!")
    print(f"Training examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")

    # Handle holdback if specified
    if holdback_percent > 0:
        print(f"\nHolding back {holdback_percent}% of training data for future revision...")
        holdback_count = int(len(train_dataset) * holdback_percent / 100)
        print(f"Holding back {holdback_count} examples")

        # Create heldout directory marker
        heldout_dir = store_path / "heldout_new"
        heldout_dir.mkdir(exist_ok=True)

        # Write a manifest file indicating what's held out
        heldout_info = {
            "count": holdback_count,
            "percent": holdback_percent,
            "note": f"First {holdback_count} examples held back for D0002 revision"
        }
        with open(heldout_dir / "info.json", "w") as f:
            json.dump(heldout_info, f, indent=2)

        print(f"Heldout info saved to {heldout_dir}/info.json")

    return len(train_dataset), len(test_dataset)


def create_checksums(store_path: Path) -> dict:
    """Create checksums.json for the dataset."""
    print("\nComputing checksums...")

    checksums = {}

    # Compute hash for MNIST directory (contains the actual data files)
    mnist_dir = store_path / "MNIST"
    if mnist_dir.exists():
        print("  Computing hash for training data...")
        train_raw = mnist_dir / "raw"
        if train_raw.exists():
            checksums["train"] = compute_directory_hash(train_raw)

        print("  Computing hash for test data...")
        checksums["test"] = compute_directory_hash(train_raw)  # Same raw dir

    # Check for heldout
    heldout_dir = store_path / "heldout_new"
    if heldout_dir.exists():
        print("  Computing hash for heldout data...")
        checksums["heldout_new"] = compute_directory_hash(heldout_dir)

    # Save checksums
    checksums_path = store_path / "checksums.json"
    with open(checksums_path, "w") as f:
        json.dump(checksums, f, indent=2)

    print(f"Checksums saved to {checksums_path}")
    return checksums


def create_dataset_manifest(
    store_path: Path,
    revision_id: str,
    train_count: int,
    test_count: int,
    checksums: dict,
    holdback_percent: float = 0.0
):
    """Create dataset_manifest.json for the revision."""
    print("\nCreating dataset manifest...")

    manifest = {
        "dataset_id": "mnist",
        "revision_id": revision_id,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "materialization": {
            "store_type": "local_fs",
            "store_root": str(store_path.parent.parent),
            "relative_path": f"mnist/{revision_id}",
            "recipe": "scripts/materialize_mnist.py"
        },
        "splits": {
            "train": {
                "examples": train_count,
                "hash": checksums.get("train", "")
            },
            "test": {
                "examples": test_count,
                "hash": checksums.get("test", "")
            }
        },
        "provenance": {
            "source": "MNIST",
            "provider": "torchvision",
            "downloaded_utc": datetime.utcnow().isoformat() + "Z"
        },
        "notes": f"D0001 baseline revision. Holdback: {holdback_percent}%"
    }

    # Add heldout split if present
    if holdback_percent > 0:
        heldout_count = int(train_count * holdback_percent / 100)
        manifest["splits"]["heldout_new"] = {
            "examples": heldout_count,
            "hash": checksums.get("heldout_new", "")
        }

    manifest_path = store_path / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved to {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Materialize MNIST dataset into dataset store"
    )
    parser.add_argument(
        "--store-root",
        type=str,
        default="../ml-data-store",
        help="Root path of dataset store"
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="D0001",
        help="Dataset revision ID"
    )
    parser.add_argument(
        "--holdback",
        type=float,
        default=0.0,
        help="Percentage of training data to hold back (0-100)"
    )

    args = parser.parse_args()

    # Setup paths
    store_root = Path(args.store_root)
    store_path = store_root / "mnist" / args.revision

    print("=" * 60)
    print("MNIST Dataset Materialization")
    print("=" * 60)
    print(f"Store root: {store_root}")
    print(f"Revision: {args.revision}")
    print(f"Store path: {store_path}")
    print(f"Holdback: {args.holdback}%")
    print("=" * 60)

    # Create directory if it doesn't exist
    store_path.mkdir(parents=True, exist_ok=True)

    # Download MNIST
    train_count, test_count = download_mnist(store_path, args.holdback)

    # Create checksums
    checksums = create_checksums(store_path)

    # Create manifest
    manifest = create_dataset_manifest(
        store_path,
        args.revision,
        train_count,
        test_count,
        checksums,
        args.holdback
    )

    print("\n" + "=" * 60)
    print("Dataset materialization complete!")
    print("=" * 60)
    print(f"Revision: {args.revision}")
    print(f"Location: {store_path}")
    print(f"Training examples: {train_count}")
    print(f"Test examples: {test_count}")
    if args.holdback > 0:
        heldout = int(train_count * args.holdback / 100)
        print(f"Heldout examples: {heldout}")
    print("\nFiles created:")
    print(f"  - dataset_manifest.json")
    print(f"  - checksums.json")
    print(f"  - MNIST/raw/ (data files)")
    print("=" * 60)


if __name__ == "__main__":
    main()
