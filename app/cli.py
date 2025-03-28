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

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.character import CharacterPersona
from src.data_processing import SourceProcessor
from src.embeddings import EmbeddingManager

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


@app.command("chat")
def chat_with_npc(
        model_path: Optional[str] = typer.Option(None, help="Path to LLM model file"),
        persona_file: Optional[str] = typer.Option(None, help="Path to character persona JSON file"),
        save_history: bool = typer.Option(True, help="Save chat history to file"),
        history_file: str = typer.Option("chat_history.json", help="File to save chat history")
):
    """Interactive chat with Thallan, the Radiant Citadel NPC."""
    global character, chat_history

    # Initialize character
    console.print("[system]Initializing Thallan...[/system]")
    character = CharacterPersona(
        name="Thallan",
        persona_file=persona_file,
        model_path=model_path
    )

    # Load existing chat history if available
    if save_history and os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                chat_history = json.load(f)
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