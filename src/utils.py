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
        model_name: str = "phi-2.Q4_K_M.gguf",
        base_url: str = "https://huggingface.co/TheBloke",
        repo_name: Optional[str] = None,
        output_dir: Optional[str] = None
) -> str:
    """
    Download an LLM model.

    Args:
        model_name: Model filename
        base_url: Base URL for download
        repo_name: Repository name, defaults to derived from model name
        output_dir: Output directory

    Returns:
        Path to downloaded model
    """
    if repo_name is None:
        # Derive repo name from model name
        if "phi-2" in model_name.lower():
            repo_name = "phi-2-GGUF"
        elif "mistral" in model_name.lower():
            repo_name = "Mistral-7B-Instruct-v0.2-GGUF"
        elif "llama" in model_name.lower():
            repo_name = "Llama-2-7B-Chat-GGUF"
        else:
            raise ValueError(f"Cannot determine repository for {model_name}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "radiant_citadel_npc",
            "models"
        )

    ensure_directory(output_dir)
    output_path = os.path.join(output_dir, model_name)

    # Check if model already exists
    if os.path.exists(output_path):
        print(f"Model already exists at {output_path}")
        return output_path

    # Format download URL
    url = f"{base_url}/{repo_name}/resolve/main/{model_name}"

    print(f"Downloading {model_name} from {url}")
    download_file(url, output_path)

    return output_path


if __name__ == "__main__":
    # Example usage
    model_path = download_llm_model()
    print(f"Downloaded model to {model_path}")