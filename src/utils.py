"""
Utility functions for the Radiant Citadel NPC project.
"""
import os
import sys
import json
import shutil
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
import torch


def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists.

    Args:
        path: Directory path

    Returns:
        The path that was created
    """
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Any, file_path: str, indent: int = 2) -> None:
    """
    Save data to a JSON file.

    Args:
        data: Data to save
        file_path: Output file path
        indent: JSON indentation
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent)


def load_json(file_path: str) -> Any:
    """
    Load data from a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def download_file(url: str, output_path: str, chunk_size: int = 1024 * 1024) -> None:
    """
    Download a file with progress bar.

    Args:
        url: URL to download
        output_path: Path to save the file
        chunk_size: Size of chunks to download at a time
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=os.path.basename(output_path)
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))


def download_llm_model(
        model_name: str = "openlm-research/open_llama_7b_v2",
        base_url: str = "https://huggingface.co",
        repo_name: Optional[str] = None,
        output_dir: Optional[str] = None,
        use_auth_token: Optional[str] = None
) -> str:
    """
    Download or access a HuggingFace model.

    Args:
        model_name: Model name or identifier
        base_url: Base URL for HuggingFace
        repo_name: Repository name, defaults to model name
        output_dir: Output directory
        use_auth_token: HuggingFace token if needed for gated models

    Returns:
        Path to model directory
    """
    if repo_name is None:
        # Map common model references to full repository paths
        model_mapping = {
            # Open-access Llama 2 variants and alternatives
            "open_llama_7b": "openlm-research/open_llama_7b_v2",
            "open_llama_3b": "openlm-research/open_llama_3b_v2",
            "stable_beluga": "stabilityai/StableBeluga-7B",
            "nous_hermes": "NousResearch/Nous-Hermes-Llama2-13b",
            "wizard_vicuna": "junelee/wizard-vicuna-13b",
            "tiny_llama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "redpajama": "togethercomputer/RedPajama-INCITE-7B-Chat",

            # Original models (require auth)
            "llama-2-7b": "meta-llama/Llama-2-7b",
            "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat",
            "phi-2": "microsoft/phi-2",
        }

        # Check if the model is in our mapping
        repo_name = model_mapping.get(model_name.lower(), model_name)

    # Set output directory
    if output_dir is None:
        cache_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "radiant_citadel_npc",
            "models"
        )
        output_dir = os.path.join(cache_dir, model_name.replace("/", "_"))

    ensure_directory(output_dir)

    print(f"Model will be downloaded to or loaded from {output_dir}")

    # The model will be downloaded automatically when loaded with Transformers
    return output_dir


def get_recommended_models():
    """
    Get a list of recommended open-access models.

    Returns:
        Dict of model types and their recommended models
    """
    return {
        "Open-access Llama derivatives (7B)": [
            "openlm-research/open_llama_7b_v2",
            "stabilityai/StableBeluga-7B",
            "togethercomputer/RedPajama-INCITE-7B-Chat"
        ],
        "Smaller models (good for lower VRAM)": [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "openlm-research/open_llama_3b_v2",
            "microsoft/phi-2"  # 2.7B parameters
        ],
        "Larger models (better quality, need more VRAM)": [
            "NousResearch/Nous-Hermes-Llama2-13b",
            "junelee/wizard-vicuna-13b"
        ]
    }


def get_optimal_device():
    """
    Determine the optimal device to use (CUDA, MPS, or CPU).

    Returns:
        The device to use for model loading
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  # For Apple Silicon
    else:
        return torch.device("cpu")


if __name__ == "__main__":
    # Example usage
    model_path = download_llm_model()
    print(f"Model path: {model_path}")

    device = get_optimal_device()
    print(f"Optimal device: {device}")

    # Print recommended models
    models = get_recommended_models()
    print("\nRecommended open-access models:")
    for category, model_list in models.items():
        print(f"\n{category}:")
        for model in model_list:
            print(f"  - {model}")