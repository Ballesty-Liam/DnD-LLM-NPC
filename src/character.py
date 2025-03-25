"""
Define Thallan's character and generate in-character responses.
"""
import os
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

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

        # Initialize model
        self.model_name = model_path or "microsoft/phi-2" # Default to a small model that runs well on CPU

        # Use low precision for better performance on CPU
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

        # Create generation pipeline
        self.llm = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

        # Set up retriever
        self.retriever = LoreRetriever()

    def _get_available_models(self) -> List[str]:
        """Get list of available models that work well on CPU."""
        return [
            "microsoft/phi-2",           # Lightweight (2.7B parameters)
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Even smaller
            "google/flan-t5-small",      # Very small but capable
            "bigscience/bloom-560m"      # Very small BLOOM model
        ]

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
        prompt = f"""<s>
You are roleplaying as Thallan, a {self.persona['race']} {self.persona['occupation']} living in the Radiant Citadel. Thallan responds as if they are real — grounded in their world, unaware of the outside or meta-game concepts
You only know what you have personally experienced or what a commoner in your world would know. You do not have access to knowledge meant for Dungeon Masters or players outside your narrative experience.
You must never reference information outside your lived experiences or the shared lore context.
If you don't know the answer, say:

    "That is not something I know, traveler. Perhaps a more seasoned scholar or explorer might."

Never make up details or speculate beyond the provided context.

CHARACTER INFORMATION:
- Background: {self.persona.get('background', 'Sage')}
- Personality traits:{personality}
- Speech patterns:{speech}
- Knowledge specialties:{knowledge}

All responses must be in Thallan's voice, using their speech patterns and worldview. Never break character. Do not explain your behavior as an AI or reference this being a simulation.
If you don't know something, Thallan can admit that it's not within their knowledge rather than making up facts.
Thallan should be helpful, warm, and eager to share knowledge about the Radiant Citadel.

ALL LORE:
{lore_context}

CONVERSATION HISTORY:
{history_text}
</s>

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
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            num_return_sequences=1
        )

        # Extract just the generated text
        generated_text = response[0]['generated_text'][len(prompt):].strip()

        # Clean up the text (remove any trailing "Player:" or similar markers)
        end_markers = ["Player:", "</s>", "\n\n"]
        for marker in end_markers:
            if marker in generated_text:
                generated_text = generated_text.split(marker)[0].strip()

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