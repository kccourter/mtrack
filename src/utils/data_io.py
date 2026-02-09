"""
Dataset I/O utilities for mtrack.
Handles dataset loading, manifest reading, and pointer management.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import hashlib


def load_dataset_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load dataset manifest from dataset_manifest.json."""
    with open(manifest_path, "r") as f:
        return json.load(f)


def get_dataset_info(
    dataset_id: str,
    revision_id: str,
    store_root: Path,
    dataset_meta_commit: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get dataset information for experiment manifest.

    Args:
        dataset_id: Dataset identifier (e.g., "mnist")
        revision_id: Dataset revision (e.g., "D0001")
        store_root: Root path of dataset store
        dataset_meta_commit: Git commit SHA from dataset meta repo

    Returns:
        Dictionary with dataset metadata
    """
    relative_path = f"{dataset_id}/{revision_id}"
    dataset_path = store_root / relative_path

    info = {
        "dataset_id": dataset_id,
        "revision_id": revision_id,
        "store": {
            "type": "local_fs",
            "root": str(store_root),
            "relative_path": relative_path
        }
    }

    if dataset_meta_commit:
        info["dataset_meta_commit"] = dataset_meta_commit

    # Try to load checksums if available
    checksums_path = dataset_path / "checksums.json"
    if checksums_path.exists():
        with open(checksums_path, "r") as f:
            info["hashes"] = json.load(f)

    return info


def write_dataset_pointer(
    experiment_dir: Path,
    dataset_id: str,
    revision_id: str,
    store_root: Path,
    checksums: Optional[Dict[str, str]] = None,
    dataset_meta_commit: Optional[str] = None
):
    """
    Write dataset pointer file for experiment.

    Args:
        experiment_dir: Experiment directory path
        dataset_id: Dataset identifier
        revision_id: Dataset revision
        store_root: Dataset store root path
        checksums: Optional checksums dictionary
        dataset_meta_commit: Optional git commit from dataset meta repo
    """
    pointer_path = experiment_dir / "pointers" / "dataset_pointer.txt"

    with open(pointer_path, "w") as f:
        f.write(f"dataset_id={dataset_id}\n")
        f.write(f"dataset_revision={revision_id}\n")
        f.write(f"data_store_root={store_root}\n")
        f.write(f"relative_path={dataset_id}/{revision_id}\n")

        if checksums:
            for split, checksum in checksums.items():
                f.write(f"checksum_{split}={checksum}\n")

        if dataset_meta_commit:
            f.write(f"dataset_meta_commit={dataset_meta_commit}\n")

    print(f"Dataset pointer saved to {pointer_path}")


def load_mnist_dataset(data_path: Path, split: str = "train"):
    """
    Load MNIST dataset from local store.

    Args:
        data_path: Path to dataset revision directory
        split: Dataset split ("train" or "test")

    Returns:
        Dataset object

    Note:
        This is a placeholder. Actual implementation depends on how
        MNIST data is stored (torchvision, numpy arrays, etc.)
    """
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        is_train = split == "train"
        dataset = datasets.MNIST(
            root=data_path.parent,
            train=is_train,
            download=True,
            transform=transform
        )

        return dataset

    except ImportError:
        raise ImportError(
            "torchvision not installed. "
            "Install with: pip install torchvision"
        )


def compute_dataset_checksums(data_path: Path, splits: list) -> Dict[str, str]:
    """
    Compute checksums for dataset splits.

    Args:
        data_path: Path to dataset revision directory
        splits: List of split names to hash

    Returns:
        Dictionary mapping split names to SHA256 hashes
    """
    checksums = {}

    for split in splits:
        split_path = data_path / split
        if split_path.exists():
            # Hash the entire directory (simplified - hash directory tree)
            hasher = hashlib.sha256()

            if split_path.is_file():
                with open(split_path, "rb") as f:
                    hasher.update(f.read())
            else:
                # For directories, hash all files in sorted order
                for file_path in sorted(split_path.rglob("*")):
                    if file_path.is_file():
                        with open(file_path, "rb") as f:
                            hasher.update(f.read())

            checksums[split] = f"sha256:{hasher.hexdigest()}"

    return checksums
