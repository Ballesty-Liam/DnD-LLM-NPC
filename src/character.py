"""
Define Thallan's character and generate in-character responses.
"""
import os
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from llama_cpp import Llama

from .retrieval import LoreRetriever


class CharacterPersona:
    """Define a D&D NPC's personality and knowledge."""

    def __init__(
            self,
            name: str = "Thallan",
            persona_file: Optional[str] = None,
            model_path: Optional[str] = None
    ):
        """
        Initialize the character persona.

        Args:
            name: Character name
            persona_file: Path to JSON file with character details
            model_path: Path to LLM model file
        """
        self.name = name

        # Load persona data
        if persona_file and os.path.exists(persona_file):
            with open(persona_file, 'r', encoding='utf-8') as f:
                self.persona = json.load(f)
        else:
            # Default Thallan persona
            self.persona = {
                "name": "Thallan",
                "race": "Half-Elf",
                "occupation": "Scholar and guide at the Radiant Citadel",
                "background": "Sage",
                "personality": [
                    "Knowledgeable and scholarly",
                    "Warm and welcoming to visitors",
                    "Speaks with occasional flowery language",
                    "Proud of the Radiant Citadel's diversity",
                    "Fascinated by the histories of different cultures"
                ],
                "speech_patterns": [
                    "Uses educational metaphors",
                    "Occasionally references obscure Citadel lore",
                    "Speaks respectfully of the Founders and Dawn Incarnates",
                    "Uses phrases like 'Indeed', 'Ah, you see', and 'In my studies...'"
                ],
                "knowledge_specialties": [
                    "Radiant Citadel history and layout",
                    "The Founders and Dawn Incarnates",
                    "The Concord and how it binds the Citadel together",
                    "The various Founder populations and their histories"
                ]
            }

        # Load or download LLM model
        self.model_path = model_path or self._get_default_model()

        # Initialize model with context window large enough for our prompts
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=4096,  # Context window size
            n_threads=4  # Adjust based on your CPU
        )

        # Set up retriever
        self.retriever = LoreRetriever()

    def _get_default_model(self) -> str:
        """Get default model path, downloading if needed."""
        # Default to a 7B parameter model that can run on CPU
        default_path = os.path.join(
            os.path.expanduser("~"),
            ".cache",
            "radiant_citadel_npc",
            "models",
            "phi-2.Q4_K_M.gguf"
        )

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(default_path), exist_ok=True)

        # Check if model exists, if not give instructions to download
        if not os.path.exists(default_path):
            print(f"Model not found at {default_path}")
            print("Please download a GGUF format model file. Example options:")
            print("- phi-2.Q4_K_M.gguf (recommended for CPU-only)")
            print("- mistral-7b-instruct-v0.2.Q4_K_M.gguf")
            print("- openhermes-2.5-mistral-7b.Q4_K_M.gguf")
            print("\nDownload from: https://huggingface.co/TheBloke")
            print(f"Save to: {default_path}")

            import sys
            sys.exit(1)

        return default_path

    def _build_prompt(self, user_input: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Build a prompt for the LLM with character persona and relevant lore.

        Args:
            user_input: Current user message
            chat_history: Previous conversation turns

        Returns:
            Formatted prompt for LLM
        """
        # Retrieve relevant lore
        lore_context = self.retriever.get_condensed_context(user_input)

        # Format character description
        personality = "\n- ".join([""] + self.persona.get("personality", []))
        speech = "\n- ".join([""] + self.persona.get("speech_patterns", []))
        knowledge = "\n- ".join([""] + self.persona.get("knowledge_specialties", []))

        # Format chat history
        history_text = ""
        if chat_history:
            for turn in chat_history[-5:]:  # Include only last 5 turns to save context
                history_text += f"Player: {turn['user']}\n"
                history_text += f"Thallan: {turn['character']}\n\n"

        # Build the full prompt
        prompt = f"""<SYSTEM>
You are roleplaying as {self.persona['name']}, a {self.persona['race']} {self.persona['occupation']} in the D&D setting of the Radiant Citadel.

CHARACTER INFORMATION:
- Background: {self.persona.get('background', 'Sage')}
- Personality traits:{personality}
- Speech patterns:{speech}
- Knowledge specialties:{knowledge}

When answering, always stay in character as Thallan. Only use information from the provided lore context.
If you don't know something, Thallan can admit that it's not within their knowledge rather than making up facts.
Thallan should be helpful, warm, and eager to share knowledge about the Radiant Citadel.

RELEVANT LORE:
{lore_context}

CONVERSATION HISTORY:
{history_text}
</SYSTEM>

Player: {user_input}

Thallan:"""

        return prompt

    def generate_response(
            self,
            user_input: str,
            chat_history: Optional[List[Dict[str, str]]] = None,
            max_tokens: int = 512,
            temperature: float = 0.7
    ) -> str:
        """
        Generate an in-character response to user input.

        Args:
            user_input: User's message
            chat_history: Previous conversation turns
            max_tokens: Maximum tokens in response
            temperature: Randomness of response (0-1)

        Returns:
            In-character response
        """
        chat_history = chat_history or []

        # Build the prompt
        prompt = self._build_prompt(user_input, chat_history)

        # Generate response
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["Player:", "</SYSTEM>"]
        )

        # Extract just the generated text
        generated_text = response["choices"][0]["text"].strip()

        return generated_text


if __name__ == "__main__":
    # Example usage
    thallan = CharacterPersona()

    # Test with some questions
    test_questions = [
        "What is the Radiant Citadel?",
        "Can you tell me about the Dawn Incarnates?",
        "What can I find in the Amaranthine Market?",
    ]

    chat_history = []

    for question in test_questions:
        print(f"Player: {question}")
        response = thallan.generate_response(question, chat_history)
        print(f"Thallan: {response}\n")

        # Update chat history
        chat_history.append({
            "user": question,
            "character": response
        })