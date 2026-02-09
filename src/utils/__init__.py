"""Utility modules for mtrack experiment tracking."""

from .stamping import (
    get_git_info,
    get_next_experiment_id,
    create_experiment_folder,
    hash_file,
    hash_dict,
    create_manifest,
    save_manifest,
    save_metrics
)

from .hf_io import (
    load_model_ref,
    get_model_info,
    load_model,
    download_model_card
)

from .data_io import (
    load_dataset_manifest,
    get_dataset_info,
    write_dataset_pointer,
    load_mnist_dataset,
    compute_dataset_checksums
)

__all__ = [
    # stamping
    "get_git_info",
    "get_next_experiment_id",
    "create_experiment_folder",
    "hash_file",
    "hash_dict",
    "create_manifest",
    "save_manifest",
    "save_metrics",
    # hf_io
    "load_model_ref",
    "get_model_info",
    "load_model",
    "download_model_card",
    # data_io
    "load_dataset_manifest",
    "get_dataset_info",
    "write_dataset_pointer",
    "load_mnist_dataset",
    "compute_dataset_checksums",
]
