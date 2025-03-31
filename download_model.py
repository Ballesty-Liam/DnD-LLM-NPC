"""
Script to download LLM models for the Radiant Citadel NPC project.
"""
import os
import sys
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import download_llm_model, get_optimal_device, get_recommended_models


def main():
    """Main function to download and verify access to the model."""
    parser = argparse.ArgumentParser(description="Download LLM model for Thallan NPC")

    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/phi-2",
        help="Model name on Hugging Face (default: microsoft/phi-2)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the model"
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model works by loading tokenizer and model"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List recommended open-access models"
    )

    args = parser.parse_args()

    # List recommended models if requested
    if args.list_models:
        models = get_recommended_models()
        print("\nRecommended open-access models:")
        for category, model_list in models.items():
            print(f"\n{category}:")
            for model in model_list:
                print(f"  - {model}")
        return

    # Get model cache path
    print(f"Preparing to download/access {args.model}...")
    model_path = download_llm_model(
        model_name=args.model,
        output_dir=args.output_dir
    )

    print(f"Model path: {model_path}")

    # Verify the model if requested
    if args.verify:
        device = get_optimal_device()
        print(f"Verifying model on {device}...")

        # Test loading tokenizer
        print("Loading tokenizer...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                print("Set padding token to EOS token")
            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            print("\nConsider trying one of these more compatible models:")
            print("  - microsoft/phi-2")
            print("  - TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            print("  - google/gemma-2b-it")
            sys.exit(1)

        # For GPU, use 4-bit quantization
        quantization_config = None
        if device.type == "cuda":
            print("Using 4-bit quantization for GPU acceleration")
            try:
                import bitsandbytes as bnb
                print(f"Found bitsandbytes version: {bnb.__version__}")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            except ImportError:
                print("Warning: bitsandbytes not installed. 4-bit quantization not available.")
                print("To enable 4-bit quantization, install: pip install bitsandbytes")

        # Test loading model
        print("Loading model (this might take a few minutes)...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print(f"✓ Successfully verified access to {args.model}")

            # Print model stats
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Model parameters: {param_count/1e9:.2f}B")
            print(f"Model device: {next(model.parameters()).device}")

            # Test if model is actually on GPU if CUDA is available
            if device.type == "cuda" and not next(model.parameters()).is_cuda:
                print("Warning: Model loaded but not on GPU. Check CUDA setup.")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("\nTry a different model that's more compatible with your hardware.")
            print("Run with --list-models to see options.")
            sys.exit(1)

    print("\nYou can now use this model with Thallan NPC by running:")
    print(f"python app/cli.py chat --model-path {args.model}")


if __name__ == "__main__":
    main()