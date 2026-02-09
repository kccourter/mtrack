"""
Hugging Face model I/O utilities for mtrack.
Handles loading models and capturing revision information.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import json


def load_model_ref(model_ref_path: Path) -> Dict[str, Any]:
    """Load model reference from model_ref.json."""
    with open(model_ref_path, "r") as f:
        return json.load(f)


def get_model_info(model_id: str, revision: str = "main") -> Dict[str, Any]:
    """
    Get model information for manifest.

    Args:
        model_id: Hugging Face model ID (e.g., "lane99/resnet_mnist_digits")
        revision: Model revision (commit SHA, tag, or branch name)

    Returns:
        Dictionary with model metadata
    """
    return {
        "provider": "huggingface",
        "model_id": model_id,
        "revision": revision
    }


def load_model(model_id: str, revision: str = "main", device: str = "cpu"):
    """
    Load a model from Hugging Face Hub.

    Args:
        model_id: Hugging Face model ID
        revision: Model revision
        device: Device to load model on

    Returns:
        Loaded model

    Note:
        This is a placeholder. Actual implementation depends on the model type
        and framework (PyTorch, TensorFlow, etc.)
    """
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True
        )
        model.to(device)
        return model
    except ImportError:
        raise ImportError(
            "transformers library not installed. "
            "Install with: pip install transformers"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {e}")


def download_model_card(model_id: str, output_path: Path):
    """
    Download model card from Hugging Face Hub.

    Args:
        model_id: Hugging Face model ID
        output_path: Where to save the model card
    """
    try:
        from huggingface_hub import hf_hub_download

        card_path = hf_hub_download(
            repo_id=model_id,
            filename="README.md",
            repo_type="model"
        )

        # Copy to output location
        with open(card_path, "r") as src:
            with open(output_path, "w") as dst:
                dst.write(src.read())

        print(f"Model card downloaded to {output_path}")
    except ImportError:
        print("huggingface_hub library not installed.")
        print("Install with: pip install huggingface-hub")
    except Exception as e:
        print(f"Failed to download model card: {e}")
        print("Using template model card instead.")
