"""
Script to download LLM models for the Radiant Citadel NPC project.
"""
import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import download_llm_model


def main():
    """Main function to download the model."""
    parser = argparse.ArgumentParser(description="Download LLM model for Thallan NPC")

    parser.add_argument(
        "--model",
        type=str,
        default="phi-2.Q4_K_M.gguf",
        help="Model filename to download"
    )

    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Repository name on Hugging Face"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the model"
    )

    args = parser.parse_args()

    print(f"Downloading {args.model}...")
    model_path = download_llm_model(
        model_name=args.model,
        repo_name=args.repo,
        output_dir=args.output_dir
    )

    print(f"Model downloaded to: {model_path}")
    print("You can now use this model with Thallan NPC")


if __name__ == "__main__":
    main()