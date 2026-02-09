"""
Artifact stamping utilities for mtrack experiments.
Generates experiment IDs, captures git state, and creates manifests.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib


def get_git_info() -> Dict[str, Any]:
    """Capture current git state."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        # Check for uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL
        ).decode().strip()

        dirty = len(status) > 0

        return {
            "commit": commit,
            "branch": branch,
            "dirty_worktree": dirty
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit": "unknown",
            "branch": "unknown",
            "dirty_worktree": True
        }


def get_next_experiment_id(experiments_dir: Path) -> str:
    """
    Find the next available experiment ID.
    Scans existing experiment folders and returns E#### with next number.
    """
    max_id = 0

    if experiments_dir.exists():
        for folder in experiments_dir.iterdir():
            if folder.is_dir() and "__E" in folder.name:
                # Extract E#### from folder name
                parts = folder.name.split("__E")
                if len(parts) == 2:
                    try:
                        exp_num = int(parts[1])
                        max_id = max(max_id, exp_num)
                    except ValueError:
                        continue

    return f"E{max_id + 1:04d}"


def create_experiment_folder(
    experiments_dir: Path,
    dataset_name: str,
    model_name: str,
    experiment_id: Optional[str] = None
) -> tuple[Path, str]:
    """
    Create experiment folder with standard naming convention.
    Returns (folder_path, experiment_id)
    """
    if experiment_id is None:
        experiment_id = get_next_experiment_id(experiments_dir)

    timestamp = datetime.utcnow().strftime("%Y-%m-%d")
    folder_name = f"{timestamp}__{dataset_name}__{model_name}__{experiment_id}"

    folder_path = experiments_dir / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (folder_path / "artifacts").mkdir(exist_ok=True)
    (folder_path / "pointers").mkdir(exist_ok=True)

    return folder_path, experiment_id


def hash_file(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return f"sha256:{sha256.hexdigest()}"


def hash_dict(data: Dict[str, Any]) -> str:
    """Compute SHA256 hash of a dictionary (via JSON serialization)."""
    json_str = json.dumps(data, sort_keys=True)
    return f"sha256:{hashlib.sha256(json_str.encode()).hexdigest()}"


def create_manifest(
    experiment_id: str,
    model_info: Dict[str, Any],
    dataset_info: Dict[str, Any],
    config_info: Dict[str, Any],
    output_paths: Dict[str, str]
) -> Dict[str, Any]:
    """
    Create experiment manifest with complete provenance.
    """
    git_info = get_git_info()

    manifest = {
        "experiment_id": experiment_id,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "git": git_info,
        "inputs": {
            "model": model_info,
            "dataset": dataset_info,
            "config": config_info
        },
        "outputs": output_paths
    }

    return manifest


def save_manifest(manifest: Dict[str, Any], experiment_dir: Path):
    """Save manifest.json to experiment directory."""
    manifest_path = experiment_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")


def save_metrics(metrics: Dict[str, Any], experiment_dir: Path):
    """Save metrics.json to experiment directory."""
    metrics_path = experiment_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
