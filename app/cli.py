"""
Command-line interface for interacting with Thallan NPC.
"""
import os
import sys
import json
from typing import List, Dict, Optional
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme
from rich.table import Table
import torch

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.character import CharacterPersona
from src.data_processing import SourceProcessor
from src.embeddings import EmbeddingManager
from src.utils import get_recommended_models, get_optimal_device

# Set up Rich theming
custom_theme = Theme({
    "player": "bold green",
    "npc": "bold yellow",
    "system": "bold blue",
    "error": "bold red"
})

console = Console(theme=custom_theme)
app = typer.Typer()

# Global chat history
chat_history: List[Dict[str, str]] = []
character: Optional[CharacterPersona] = None

# Development mode for debugging
DEV_MODE = False


@app.command("process")
def process_source_material(
        input_dir: str = typer.Option("data/raw", help="Directory with source markdown files"),
        output_file: str = typer.Option("chunks.jsonl", help="Output filename for processed chunks"),
        chunk_size: int = typer.Option(500, help="Size of text chunks")
):
    """Process D&D source material into chunks for embedding."""
    console.print("[system]Processing source material...[/system]")

    processor = SourceProcessor(chunk_size=chunk_size, raw_dir=input_dir)
    documents = processor.process_and_save(output_file=output_file)

    console.print(f"[system]✓ Processed {len(documents)} chunks from {input_dir}[/system]")


@app.command("embed")
def create_embeddings(
        input_file: str = typer.Option("chunks.jsonl", help="Input chunks file"),
        model_name: str = typer.Option("all-MiniLM-L6-v2", help="Embedding model name")
):
    """Create embeddings for processed chunks."""
    console.print("[system]Creating embeddings...[/system]")

    manager = EmbeddingManager(model_name=model_name)
    manager.process_and_embed(chunks_file=input_file)

    console.print(f"[system]✓ Created embeddings using {model_name}[/system]")


@app.command("models")
def list_models():
    """List recommended open-access models."""
    models = get_recommended_models()

    console.print("[system]Recommended open-access models for Thallan NPC:[/system]")

    # Get device info
    device = get_optimal_device()
    device_info = f"Currently using: {device.type}"
    if device.type == "cuda":
        device_info += f" ({torch.cuda.get_device_name(0)})"

    console.print(f"[system]{device_info}[/system]")

    for category, model_list in models.items():
        table = Table(title=category)
        table.add_column("Model Name", style="cyan")
        table.add_column("Description", style="green")

        for model in model_list:
            description = ""
            if "tinyllama" in model.lower():
                description = "Small (1.1B) but efficient model, great compatibility"
            elif "phi" in model.lower():
                description = "Microsoft's small but capable model, excellent on lower-end hardware"
            elif "gemma" in model.lower():
                description = "Google's lightweight model, good for most hardware"
            elif "stable" in model.lower():
                description = "Stability AI's model with good instruction following"
            elif "nous" in model.lower():
                description = "High-quality but requires significant GPU memory"

            table.add_row(model, description)

        console.print(table)
        console.print()

    console.print("[system]Usage: python app/cli.py chat --model-path MODEL_NAME --force-gpu[/system]")
    console.print("[system]Example: python app/cli.py chat --model-path microsoft/phi-2 --force-gpu[/system]")


@app.command("gpu-info")
def gpu_info():
    """Display detailed GPU information."""
    if not torch.cuda.is_available():
        console.print("[error]No CUDA-capable GPU detected[/error]")
        console.print("[system]Check your PyTorch installation by running: pip list | findstr torch[/system]")
        console.print("[system]If you see +cpu, reinstall with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121[/system]")
        return

    console.print("[system]GPU Information:[/system]")

    table = Table(title="CUDA Device Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    device_count = torch.cuda.device_count()
    current_device = torch.cuda.current_device()

    table.add_row("CUDA Available", "Yes")
    table.add_row("Device Count", str(device_count))
    table.add_row("Current Device", str(current_device))
    table.add_row("Device Name", torch.cuda.get_device_name(current_device))

    props = torch.cuda.get_device_properties(current_device)
    table.add_row("Compute Capability", f"{props.major}.{props.minor}")
    table.add_row("Total Memory", f"{props.total_memory / 1e9:.2f} GB")
    table.add_row("Multi-Processor Count", str(props.multi_processor_count))

    # Memory information
    table.add_row("Memory Allocated", f"{torch.cuda.memory_allocated() / 1e9:.2f} GB")
    table.add_row("Memory Reserved", f"{torch.cuda.memory_reserved() / 1e9:.2f} GB")
    table.add_row("Max Memory Allocated", f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    console.print(table)

    # Additional info about PyTorch
    console.print("\n[system]PyTorch CUDA Information:[/system]")
    console.print(f"PyTorch version: {torch.__version__}")
    console.print(f"CUDA version: {torch.version.cuda}")


@app.command("dev-mode")
def toggle_dev_mode(enable: bool = True):
    """Enable or disable development mode with verbose output."""
    global DEV_MODE
    DEV_MODE = enable
    if enable:
        console.print("[system]Development mode enabled - verbose diagnostic output will be shown[/system]")
    else:
        console.print("[system]Development mode disabled - diagnostic output suppressed[/system]")


@app.command("chat")
def chat_with_npc(
        model_path: str = typer.Option("microsoft/phi-2", help="Path or name of LLM model"),
        persona_file: Optional[str] = typer.Option(None, help="Path to character persona JSON file"),
        save_history: bool = typer.Option(True, help="Save chat history to file"),
        history_file: str = typer.Option("chat_history.json", help="File to save chat history"),
        force_gpu: bool = typer.Option(False, help="Force GPU acceleration"),
        strict_knowledge: bool = typer.Option(True, help="Strictly enforce knowledge limitations"),
        dev_mode: bool = typer.Option(False, help="Enable development mode with verbose output")
):
    """Interactive chat with Thallan, the Radiant Citadel NPC."""
    global character, chat_history, DEV_MODE

    # Update dev mode from parameter
    DEV_MODE = dev_mode

    # Show GPU info if forcing GPU
    if force_gpu and DEV_MODE:
        if torch.cuda.is_available():
            console.print(f"[system]GPU acceleration enabled: {torch.cuda.get_device_name(0)}[/system]")
            # Clear CUDA cache
            torch.cuda.empty_cache()
            console.print(f"[system]Cleared GPU cache[/system]")
        else:
            console.print("[error]Cannot use GPU: No CUDA-capable GPU detected[/error]")
            console.print("[system]Make sure you have PyTorch with CUDA installed:[/system]")
            console.print("[system]pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121[/system]")
            return

    # Show anti-hallucination status in dev mode
    if DEV_MODE:
        if strict_knowledge:
            console.print("[system]Anti-hallucination system enabled - Thallan will only use information from the source material[/system]")
        else:
            console.print("[system]WARNING: Anti-hallucination system disabled - Thallan may invent information not in source material[/system]")

    # Initialize character
    console.print(f"[system]Initializing Thallan...[/system]")
    try:
        character = CharacterPersona(
            name="Thallan",
            persona_file=persona_file,
            model_path=model_path,
            force_gpu=force_gpu,
            strict_knowledge=strict_knowledge,
            verbose=DEV_MODE
        )
    except Exception as e:
        console.print(f"[error]Error initializing model: {e}[/error]")
        console.print("[system]Try running 'python app/cli.py models' to see compatible models[/system]")
        return

    # Load existing chat history if available
    if save_history and os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
            if DEV_MODE:
                console.print(f"[system]Loaded {len(chat_history)} previous chat messages[/system]")
        except Exception as e:
            console.print(f"[error]Error loading chat history: {e}[/error]")

    # Display welcome message
    console.print(Panel.fit(
        "[npc]Greetings, traveler! I am Thallan, a scholar and guide of the Radiant Citadel. "
        "How may I assist you on your journey today?[/npc]",
        title="Thallan",
        border_style="yellow"
    ))

    # Main chat loop
    try:
        while True:
            # Get user input
            user_input = Prompt.ask("[player]You[/player]")

            if user_input.lower() in ("exit", "quit", "bye", "/quit", "/exit"):
                console.print("[npc]Farewell, traveler! May the light of the Citadel guide your path.[/npc]")
                break

            # Add commands for controlling the anti-hallucination system and dev mode
            if user_input.lower() == "/strict on":
                character.strict_knowledge = True
                console.print("[system]Anti-hallucination system enabled[/system]")
                continue

            if user_input.lower() == "/strict off":
                character.strict_knowledge = False
                console.print("[system]Anti-hallucination system disabled[/system]")
                continue

            if user_input.lower() == "/dev on":
                DEV_MODE = True
                character.verbose = True
                console.print("[system]Development mode enabled - verbose output will be shown[/system]")
                continue

            if user_input.lower() == "/dev off":
                DEV_MODE = False
                character.verbose = False
                console.print("[system]Development mode disabled - verbose output suppressed[/system]")
                continue

            if user_input.lower() == "/help":
                console.print("[system]Available commands:[/system]")
                console.print("[system]  /strict on - Enable anti-hallucination system[/system]")
                console.print("[system]  /strict off - Disable anti-hallucination system[/system]")
                console.print("[system]  /dev on - Enable development mode with diagnostic output[/system]")
                console.print("[system]  /dev off - Disable development mode[/system]")
                console.print("[system]  /quit or /exit - End the conversation[/system]")
                console.print("[system]  /help - Show this help message[/system]")
                continue

            # Generate response
            with console.status("[system]Thallan is thinking...[/system]"):
                response = character.generate_response(user_input, chat_history)

            # Display response
            console.print(Panel(
                Markdown(response),
                title="Thallan",
                border_style="yellow"
            ))

            # Update chat history
            chat_history.append({
                "user": user_input,
                "character": response
            })

            # Save chat history
            if save_history:
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(chat_history, f, indent=2)

    except KeyboardInterrupt:
        console.print("\n[npc]Farewell, traveler! Until we meet again.[/npc]")
    except Exception as e:
        console.print(f"[error]An error occurred: {e}[/error]")


@app.command("setup")
def setup():
    """Complete setup process: process source, create embeddings, and chat."""
    process_source_material()
    create_embeddings()
    chat_with_npc()


if __name__ == "__main__":
    app()