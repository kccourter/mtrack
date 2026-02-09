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


def run_training(config: dict):
    """
    Run training with the given configuration.

    This is a placeholder for actual training logic.
    Implement based on your specific model and framework.
    """
    print("\n=== Training ===")
    print("Training logic not yet implemented.")
    print("This is where you would:")
    print("  1. Load the model from Hugging Face")
    print("  2. Load the dataset")
    print("  3. Run training loop")
    print("  4. Save checkpoints")
    print("\nFor now, returning mock results.")

    # Mock results
    results = {
        "primary_metric": {"name": "accuracy", "value": 0.984},
        "loss": 0.041,
        "eval": {
            "accuracy": 0.984,
            "precision_macro": 0.984,
            "recall_macro": 0.983
        }
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
    metrics = run_training(resolved_config)

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
