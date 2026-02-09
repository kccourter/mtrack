"""
Evaluation script for mtrack experiments.
Can be run standalone or as part of training.
"""

import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_manifest(experiment_dir: Path) -> dict:
    """Load experiment manifest."""
    manifest_path = experiment_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        return json.load(f)


def run_evaluation(experiment_dir: Path):
    """
    Run evaluation on a trained model from an experiment.

    This is a placeholder for actual evaluation logic.
    """
    print(f"\n=== Evaluating Experiment: {experiment_dir.name} ===")

    # Load manifest to get model and dataset info
    manifest = load_manifest(experiment_dir)

    print(f"Model: {manifest['inputs']['model']['model_id']}")
    print(f"Dataset: {manifest['inputs']['dataset']['dataset_id']}")
    print(f"Dataset Revision: {manifest['inputs']['dataset']['revision_id']}")

    print("\nEvaluation logic not yet implemented.")
    print("This is where you would:")
    print("  1. Load the trained model/checkpoint")
    print("  2. Load the evaluation dataset")
    print("  3. Run inference")
    print("  4. Compute metrics")
    print("  5. Generate visualizations")

    # Mock evaluation results
    eval_metrics = {
        "accuracy": 0.984,
        "precision_macro": 0.984,
        "recall_macro": 0.983,
        "f1_macro": 0.983,
        "confusion_matrix": "See artifacts/confusion_matrix.png"
    }

    # Save additional evaluation metrics
    metrics_path = experiment_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["detailed_eval"] = eval_metrics

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\nUpdated metrics saved to {metrics_path}")

    return eval_metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate mtrack experiment")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Path to experiment directory"
    )

    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"Error: Experiment directory not found: {experiment_dir}")
        sys.exit(1)

    if not (experiment_dir / "manifest.json").exists():
        print(f"Error: No manifest.json found in {experiment_dir}")
        sys.exit(1)

    metrics = run_evaluation(experiment_dir)

    print("\n=== Evaluation Complete ===")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
