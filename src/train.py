"""
Training script for mtrack experiments.
Handles experiment setup, training, evaluation, and artifact stamping.
"""

import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import precision_score, recall_score, f1_score

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    create_experiment_folder,
    create_manifest,
    save_manifest,
    save_metrics,
    get_model_info,
    get_dataset_info,
    write_dataset_pointer,
    hash_dict,
)
from src.model import load_model, save_checkpoint


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_config(config: dict, trial_params: dict = None) -> dict:
    """
    Resolve configuration with trial-specific parameters.

    Args:
        config: Base configuration
        trial_params: Optional trial-specific parameters for hyperparameter scans

    Returns:
        Resolved configuration dictionary
    """
    resolved = config.copy()

    if trial_params:
        # Merge trial parameters into training section
        if "training" not in resolved:
            resolved["training"] = {}
        resolved["training"].update(trial_params)

    return resolved


def save_resolved_config(config: dict, experiment_dir: Path):
    """Save resolved configuration to experiment directory."""
    config_path = experiment_dir / "config_resolved.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Resolved config saved to {config_path}")
    return config_path


def create_notes_template(experiment_dir: Path, experiment_id: str):
    """Create notes.md template for the experiment."""
    notes_path = experiment_dir / "notes.md"

    template = f"""# {experiment_id} Notes

## Intent
What are we testing and why?

## Change Summary
- Model:
- Data:
- Config / scan:
- Code:

## Results
- Primary metric:
- Secondary metrics:
- Observed failure modes:

## Next Decision
Based on results, what do we do next?
"""

    with open(notes_path, "w") as f:
        f.write(template)
    print(f"Notes template created at {notes_path}")


def get_device(device_config: str = "auto"):
    """Determine the best available device."""
    if device_config == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_config

    print(f"Using device: {device}")
    return device


def load_mnist_data(dataset_store: Path, batch_size: int, num_workers: int = 4):
    """Load MNIST dataset from the dataset store."""
    print("\n=== Loading Dataset ===")

    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load training and test datasets
    train_dataset = datasets.MNIST(
        root=dataset_store / "mnist" / "D0001",
        train=True,
        download=False,  # Already downloaded
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=dataset_store / "mnist" / "D0001",
        train=False,
        download=False,
        transform=transform
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # Larger batch for eval
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, criterion, optimizer, device, log_interval=10):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        if batch_idx % log_interval == 0:
            print(f'  Batch {batch_idx}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | '
                  f'Acc: {100. * correct / total:.2f}%')

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set."""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            all_preds.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy())

    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total

    # Compute additional metrics
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')

    return avg_loss, accuracy, precision, recall, f1


def run_training(config: dict, experiment_dir: Path, dataset_store: Path):
    """
    Run training with the given configuration.

    Args:
        config: Resolved configuration dictionary
        experiment_dir: Path to experiment directory
        dataset_store: Path to dataset store root

    Returns:
        Dictionary with training and evaluation metrics
    """
    print("\n=== Training ===")

    # Extract configuration
    training_config = config.get("training", {})
    dataset_config = config.get("dataset", {})
    compute_config = config.get("compute", {})
    model_config = config.get("model", {})

    batch_size = training_config.get("batch_size", 64)
    learning_rate = training_config.get("learning_rate", 0.001)
    weight_decay = training_config.get("weight_decay", 0.0)
    dropout = training_config.get("dropout", 0.0)
    epochs = training_config.get("epochs", 10)
    seed = training_config.get("seed", 42)
    optimizer_name = training_config.get("optimizer", "adam")

    num_workers = compute_config.get("num_workers", 4)
    device_config = compute_config.get("device", "auto")
    log_interval = config.get("logging", {}).get("log_interval", 10)

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Get device
    device = get_device(device_config)

    # Load data
    train_loader, test_loader = load_mnist_data(
        dataset_store,
        batch_size,
        num_workers
    )

    # Load model
    print("\n=== Initializing Model ===")
    model = load_model(
        model_config.get("model_id"),
        num_classes=10,
        dropout=dropout,
        device=device
    )

    # Setup optimizer
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    print(f"\nOptimizer: {optimizer_name}")
    print(f"Learning rate: {learning_rate}")
    print(f"Weight decay: {weight_decay}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")

    # Training loop
    print(f"\n=== Training for {epochs} epochs ===")
    start_time = time.time()

    best_test_acc = 0.0
    train_history = []

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, log_interval
        )

        # Evaluate
        test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
            model, test_loader, criterion, device
        )

        print(f'  Train - Loss: {train_loss:.4f}, Acc: {100*train_acc:.2f}%')
        print(f'  Test  - Loss: {test_loss:.4f}, Acc: {100*test_acc:.2f}%, '
              f'Precision: {100*test_precision:.2f}%, Recall: {100*test_recall:.2f}%')

        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            checkpoint_path = experiment_dir / "artifacts" / "best_model.pt"
            save_checkpoint(model, optimizer, epoch, test_loss, checkpoint_path)

        # Record history
        train_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1
        })

    training_time = time.time() - start_time
    print(f"\n=== Training Complete ===")
    print(f"Total time: {training_time:.2f}s ({training_time/60:.2f}m)")
    print(f"Best test accuracy: {100*best_test_acc:.2f}%")

    # Final evaluation on best model
    print("\n=== Final Evaluation ===")
    best_checkpoint = experiment_dir / "artifacts" / "best_model.pt"
    if best_checkpoint.exists():
        checkpoint = torch.load(best_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    final_loss, final_acc, final_precision, final_recall, final_f1 = evaluate(
        model, test_loader, criterion, device
    )

    # Compile results
    results = {
        "primary_metric": {
            "name": "accuracy",
            "value": float(final_acc)
        },
        "loss": float(final_loss),
        "eval": {
            "accuracy": float(final_acc),
            "precision_macro": float(final_precision),
            "recall_macro": float(final_recall),
            "f1_macro": float(final_f1)
        },
        "training": {
            "epochs": epochs,
            "best_test_accuracy": float(best_test_acc),
            "training_time_seconds": training_time,
            "final_train_loss": float(train_history[-1]["train_loss"]),
            "final_train_acc": float(train_history[-1]["train_acc"])
        },
        "history": train_history
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Run mtrack training experiment")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (baseline.yaml or scan.yaml)"
    )
    parser.add_argument(
        "--experiment-id",
        type=str,
        default=None,
        help="Override experiment ID (default: auto-generate)"
    )
    parser.add_argument(
        "--dataset-store",
        type=str,
        default="/ml-data-store",
        help="Root path of dataset store"
    )

    args = parser.parse_args()

    # Setup paths
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / args.config
    experiments_dir = repo_root / "experiments"
    dataset_store = Path(args.dataset_store)

    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Extract model and dataset info
    model_id = config["model"]["model_id"]
    model_revision = config["model"].get("revision", "main")
    dataset_id = config["dataset"]["dataset_id"]
    dataset_revision = config["dataset"]["revision_id"]

    # Create experiment folder
    model_name = model_id.split("/")[-1]  # Extract short name
    experiment_dir, experiment_id = create_experiment_folder(
        experiments_dir,
        dataset_name=dataset_id,
        model_name=model_name,
        experiment_id=args.experiment_id
    )

    print(f"\n=== Experiment {experiment_id} ===")
    print(f"Experiment directory: {experiment_dir}")

    # Resolve configuration
    resolved_config = resolve_config(config)
    config_path_resolved = save_resolved_config(resolved_config, experiment_dir)

    # Create notes template
    create_notes_template(experiment_dir, experiment_id)

    # Get model and dataset info for manifest
    model_info = get_model_info(model_id, model_revision)
    dataset_info = get_dataset_info(
        dataset_id,
        dataset_revision,
        dataset_store
    )

    # Write dataset pointer
    write_dataset_pointer(
        experiment_dir,
        dataset_id,
        dataset_revision,
        dataset_store,
        checksums=dataset_info.get("hashes")
    )

    # Create config info
    config_info = {
        "path": str(config_path.relative_to(repo_root)),
        "resolved_path": f"experiments/{experiment_dir.name}/config_resolved.yaml",
        "config_hash": hash_dict(resolved_config)
    }

    # Run training
    print("\nStarting training...")
    metrics = run_training(resolved_config, experiment_dir, dataset_store)

    # Save metrics
    save_metrics(metrics, experiment_dir)

    # Create and save manifest
    output_paths = {
        "metrics_path": f"experiments/{experiment_dir.name}/metrics.json",
        "config_resolved_path": f"experiments/{experiment_dir.name}/config_resolved.yaml",
        "notes_path": f"experiments/{experiment_dir.name}/notes.md",
        "artifacts_dir": f"experiments/{experiment_dir.name}/artifacts"
    }

    manifest = create_manifest(
        experiment_id,
        model_info,
        dataset_info,
        config_info,
        output_paths
    )

    save_manifest(manifest, experiment_dir)

    print(f"\n=== Experiment {experiment_id} Complete ===")
    print(f"Results saved to: {experiment_dir}")
    print(f"Primary metric: {metrics['primary_metric']['name']} = {metrics['primary_metric']['value']}")
    print("\nNext steps:")
    print("  1. Review notes.md and add your observations")
    print("  2. Commit the experiment results to Git")
    print(f"     git add experiments/{experiment_dir.name}")
    print(f"     git commit -m 'feat: experiment {experiment_id} - initial baseline'")


if __name__ == "__main__":
    main()
