#!/usr/bin/env python3
"""
Create dataset revision D0002 by promoting held-back data from D0001.
This simulates incremental data growth in the mtrack framework.
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime
import shutil
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


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


def load_d0001_manifest(d0001_path: Path) -> dict:
    """Load D0001 manifest to understand what data was held back."""
    manifest_path = d0001_path / "dataset_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"D0001 manifest not found at {manifest_path}")

    with open(manifest_path, "r") as f:
        return json.load(f)


def create_d0002(store_root: Path):
    """
    Create D0002 by promoting held-back data from D0001.

    In D0002, the training set conceptually includes the previously
    held-back examples. Since MNIST is downloaded as a whole, we just
    update the manifest to reflect that all training data is now available.
    """
    d0001_path = store_root / "mnist" / "D0001"
    d0002_path = store_root / "mnist" / "D0002"

    print("=" * 60)
    print("Creating Dataset Revision D0002")
    print("=" * 60)
    print(f"Source (D0001): {d0001_path}")
    print(f"Target (D0002): {d0002_path}")
    print("=" * 60)

    # Load D0001 manifest
    print("\nLoading D0001 manifest...")
    d0001_manifest = load_d0001_manifest(d0001_path)

    train_examples = d0001_manifest["splits"]["train"]["examples"]
    heldout_examples = d0001_manifest["splits"]["heldout_new"]["examples"]
    test_examples = d0001_manifest["splits"]["test"]["examples"]

    print(f"D0001 had {train_examples} training examples")
    print(f"D0001 held back {heldout_examples} examples")
    print(f"D0002 will have {train_examples + heldout_examples} training examples")

    # Create D0002 directory
    d0002_path.mkdir(parents=True, exist_ok=True)

    # Copy MNIST data from D0001 (it's the same underlying data)
    print("\nCopying MNIST data from D0001...")
    if (d0001_path / "MNIST").exists():
        if (d0002_path / "MNIST").exists():
            shutil.rmtree(d0002_path / "MNIST")
        shutil.copytree(d0001_path / "MNIST", d0002_path / "MNIST")
        print("  MNIST data copied")

    # Note: In a real scenario with separate held-back files, we would
    # merge them here. For MNIST, the data is already complete in the
    # downloaded files, so we just update the metadata.

    # Compute checksums
    print("\nComputing checksums for D0002...")
    checksums = {}

    mnist_dir = d0002_path / "MNIST"
    if mnist_dir.exists():
        train_raw = mnist_dir / "raw"
        if train_raw.exists():
            checksums["train"] = compute_directory_hash(train_raw)
            checksums["test"] = compute_directory_hash(train_raw)

    # No more heldout in D0002 - it's all been promoted
    checksums["heldout_new"] = "sha256:empty"

    # Save checksums
    checksums_path = d0002_path / "checksums.json"
    with open(checksums_path, "w") as f:
        json.dump(checksums, f, indent=2)
    print(f"Checksums saved to {checksums_path}")

    # Create D0002 manifest
    print("\nCreating D0002 manifest...")

    manifest = {
        "dataset_id": "mnist",
        "revision_id": "D0002",
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "materialization": {
            "store_type": "local_fs",
            "store_root": str(store_root),
            "relative_path": "mnist/D0002",
            "recipe": "scripts/create_dataset_d0002.py",
            "parent_revision": "D0001"
        },
        "splits": {
            "train": {
                "examples": train_examples + heldout_examples,  # Promoted!
                "hash": checksums.get("train", "")
            },
            "test": {
                "examples": test_examples,
                "hash": checksums.get("test", "")
            },
            "heldout_new": {
                "examples": 0,
                "hash": checksums.get("heldout_new", ""),
                "note": "All previously held-back data promoted to training set"
            }
        },
        "provenance": {
            "source": "MNIST",
            "provider": "torchvision",
            "parent_revision": "D0001",
            "promoted_from_heldout": heldout_examples,
            "created_utc": datetime.utcnow().isoformat() + "Z"
        },
        "notes": f"D0002 revision: promoted {heldout_examples} held-back examples from D0001 into training set. " +
                 f"Total training examples: {train_examples + heldout_examples}."
    }

    manifest_path = d0002_path / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")

    # Create a notes file explaining the revision
    notes_path = d0002_path / "revision_notes.md"
    with open(notes_path, "w") as f:
        f.write(f"""# Dataset Revision D0002

## Change Summary

Promoted previously held-back data from D0001 into the training set.

## Details

- **Parent Revision:** D0001
- **Created:** {datetime.utcnow().isoformat()}Z
- **Promoted Examples:** {heldout_examples}
- **New Training Size:** {train_examples + heldout_examples} (was {train_examples})
- **Test Size:** {test_examples} (unchanged)

## Rationale

This revision simulates the arrival of "new" training data in an incremental
learning workflow. The {heldout_examples} examples that were held back in D0001
are now available for training.

## Next Steps

Run experiments with D0002 to measure:
1. Performance improvement from additional training data
2. Model behavior with dataset expansion
3. Comparison against D0001 baseline

## Usage

Update experiment configs to use `revision_id: D0002`
""")
    print(f"Revision notes saved to {notes_path}")

    print("\n" + "=" * 60)
    print("Dataset revision D0002 created successfully!")
    print("=" * 60)
    print(f"Location: {d0002_path}")
    print(f"Training examples: {train_examples + heldout_examples} (+{heldout_examples} from D0001)")
    print(f"Test examples: {test_examples}")
    print(f"Heldout examples: 0 (all promoted)")
    print("\nFiles created:")
    print("  - dataset_manifest.json")
    print("  - checksums.json")
    print("  - revision_notes.md")
    print("  - MNIST/raw/ (data files)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Create dataset revision D0002 from D0001"
    )
    parser.add_argument(
        "--store-root",
        type=str,
        default="../ml-data-store",
        help="Root path of dataset store"
    )

    args = parser.parse_args()
    store_root = Path(args.store_root)

    # Check that D0001 exists
    d0001_path = store_root / "mnist" / "D0001"
    if not d0001_path.exists():
        print(f"Error: D0001 not found at {d0001_path}")
        print("Please run materialize_mnist.py first to create D0001")
        sys.exit(1)

    create_d0002(store_root)


if __name__ == "__main__":
    main()
