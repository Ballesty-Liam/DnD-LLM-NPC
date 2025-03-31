"""
Define Thallan's character and generate in-character responses.
"""
import os
import re
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, BitsAndBytesConfig

from .retrieval import LoreRetriever
from .utils import get_optimal_device

# Import special handler from dedicated file
try:
    from .model_handlers import load_openllama_model
except ImportError:
    # Inline definition if module not available
    import torch
    def load_openllama_model(model_name, cache_dir=None, use_gpu=True):
        print(f"Using specialized loader for OpenLLaMA model: {model_name}")

        # Clear CUDA cache if using GPU
        if use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # First try loading with the standard Llama tokenizer
        try:
            # Use LlamaTokenizer instead of auto tokenizer
            from transformers import LlamaTokenizer
            tokenizer = LlamaTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                legacy=True  # This is important for older Llama models
            )
        except Exception as e:
            print(f"Error with LlamaTokenizer: {e}")

            # Fallback to a known-good Llama tokenizer
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    "huggyllama/llama-7b",
                    cache_dir=cache_dir,
                    use_fast=False
                )
            except Exception as e2:
                print(f"Error with fallback tokenizer: {e2}")

                # Final fallback to a completely different tokenizer
                from transformers import GPT2Tokenizer
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        # Set padding token if necessary
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure precision and device settings
        device_map = "auto" if (use_gpu and torch.cuda.is_available()) else None
        torch_dtype = torch.float16 if (use_gpu and torch.cuda.is_available()) else torch.float32

        # Load the actual model with proper settings
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        )

        return {"model": model, "tokenizer": tokenizer}


class CharacterPersona:
    """Define a D&D NPC's personality and knowledge."""

    def __init__(
        self,
        name: str = "Thallan",
        persona_file: Optional[str] = None,
        model_path: Optional[str] = None,
        force_gpu: bool = True,
        strict_knowledge: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the character persona.

        Args:
            name: Character name
            persona_file: Path to JSON file with character details
            model_path: Path to LLM model or model identifier
            force_gpu: Whether to force GPU usage
            strict_knowledge: Whether to strictly enforce knowledge limits
            verbose: Whether to print detailed diagnostic information
        """
        self.name = name
        self.strict_knowledge = strict_knowledge
        self.verbose = verbose

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
                    "Speaks with occasional flowery language"
                ],
                "speech_patterns": [
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

        # Get optimal device for inference
        self.device = get_optimal_device()
        if self.verbose:
            print(f"Using device: {self.device}")

        # Force GPU if available and requested
        if force_gpu and torch.cuda.is_available():
            if self.verbose:
                print("GPU acceleration enabled")
            torch.cuda.empty_cache()  # Clear GPU memory

        # Initialize model
        self.model_name = model_path or "microsoft/phi-2"

        # Configure quantization for optimal GPU performance
        quantization_config = None
        if self.device.type == "cuda":
            # Try to use 4-bit quantization
            try:
                import bitsandbytes as bnb
                if self.verbose:
                    print(f"Using bitsandbytes for quantization (version {bnb.__version__})")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
            except ImportError:
                if self.verbose:
                    print("bitsandbytes not available, using 16-bit precision instead")

        if self.verbose:
            print(f"Loading model: {self.model_name}")

        # Special handling for OpenLLaMA models
        if "open_llama" in self.model_name.lower():
            try:
                # Use specialized loader for OpenLLaMA
                result = load_openllama_model(
                    model_name=self.model_name,
                    use_gpu=(self.device.type == "cuda")
                )
                self.model = result["model"]
                self.tokenizer = result["tokenizer"]
                if self.verbose:
                    print("Successfully loaded OpenLLaMA model and tokenizer")

            except Exception as e:
                if self.verbose:
                    print(f"Error loading OpenLLaMA model: {e}")
                    print("Falling back to microsoft/phi-2 model which has better compatibility")
                self.model_name = "microsoft/phi-2"

                # Load standard tokenizer and model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )
        else:
            # Standard loading procedure for non-OpenLLaMA models
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    trust_remote_code=True
                )

                # Set padding token if not set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                if self.verbose:
                    print("Successfully loaded tokenizer")

                # Load model with GPU acceleration via device_map="auto"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",  # This handles GPU placement automatically
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )

            except Exception as e:
                if self.verbose:
                    print(f"Error loading model: {e}")
                    print("Falling back to microsoft/phi-2 model which has better compatibility")
                self.model_name = "microsoft/phi-2"
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    low_cpu_mem_usage=True
                )

        # Check if model is using GPU
        if self.device.type == "cuda" and self.verbose:
            # Check if any parameter is on CUDA
            is_on_gpu = any(p.is_cuda for p in self.model.parameters())
            if is_on_gpu:
                print("Model successfully loaded on GPU")
            else:
                print("Warning: Model parameters not on GPU despite CUDA being available")

        # Get parameter count
        if self.verbose:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Parameter count: {param_count/1e9:.2f} billion")

        # Create text generation pipeline WITHOUT specifying device (let accelerate handle it)
        try:
            self.llm = TextGenerationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False
            )
            if self.verbose:
                print("Successfully created text generation pipeline")
        except Exception as e:
            if self.verbose:
                print(f"Error creating pipeline: {e}")
            # Try an alternative approach
            self.llm = TextGenerationPipeline(
                model=self.model,
                tokenizer=self.tokenizer,
                return_full_text=False,
                device_map="auto"  # Use device_map instead of device
            )

        # Set up retriever with increased result count for better context
        self.retriever = LoreRetriever(results_count=8)  # Increased from default

    def _extract_key_entities(self, text: str) -> List[str]:
        """
        Extract key entities from text for grounding checks.
        Uses simple pattern matching for names, places, and concepts.

        Args:
            text: Text to analyze

        Returns:
            List of extracted entities
        """
        # Extract capitalized words/phrases (likely proper nouns)
        capitalized_pattern = r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b'
        capitalized = re.findall(capitalized_pattern, text)

        # Extract quoted phrases (likely specific terms or names)
        quoted_pattern = r'"([^"]*)"'
        quoted = re.findall(quoted_pattern, text)

        # Combine and remove duplicates
        entities = list(set(capitalized + quoted))

        # Filter out common words
        stopwords = ["I", "You", "They", "We", "It", "The", "A", "An", "This", "That"]
        entities = [e for e in entities if e not in stopwords and len(e) > 1]

        return entities

    def _is_response_grounded(self, response: str, context: str) -> bool:
        """
        Check if response is grounded in the provided context.

        Args:
            response: Generated response text
            context: Retrieval context text

        Returns:
            True if response appears to be grounded in context
        """
        # Extract entities from response and context
        response_entities = self._extract_key_entities(response)
        context_entities = self._extract_key_entities(context)

        # No entities found in response - likely a generic answer
        if not response_entities:
            return True

        # Find ungrounded entities (appear in response but not in context)
        ungrounded_entities = []
        for entity in response_entities:
            # Skip very short entities
            if len(entity) < 3:
                continue

            # Check if entity or similar exists in context
            found = False
            for context_entity in context_entities:
                # Exact match
                if entity == context_entity:
                    found = True
                    break
                # Partial match (entity is part of a context entity)
                if entity in context_entity or context_entity in entity:
                    found = True
                    break

            if not found:
                ungrounded_entities.append(entity)

        # Calculate ratio of ungrounded entities
        if not response_entities:
            return True

        ungrounded_ratio = len(ungrounded_entities) / len(response_entities)

        # Allow a small percentage of ungrounded entities (common words might be false positives)
        is_grounded = ungrounded_ratio <= 0.3

        if not is_grounded and self.verbose:
            print(f"Response contains ungrounded entities: {ungrounded_entities}")

        return is_grounded

    def _build_prompt(self, user_input: str, chat_history: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Build a prompt for the LLM with character persona and relevant lore.
        Works with open-access LLMs which may have different prompt formats.

        Args:
            user_input: Current user message
            chat_history: Previous conversation turns

        Returns:
            Dictionary with prompt text and relevant context
        """
        # Retrieve relevant lore with more results for better coverage
        lore_context = self.retriever.get_condensed_context(user_input, detailed=True, verbose=self.verbose)

        # Format character description
        personality = "\n- ".join([""] + self.persona.get("personality", []))
        speech = "\n- ".join([""] + self.persona.get("speech_patterns", []))
        knowledge = "\n- ".join([""] + self.persona.get("knowledge_specialties", []))

        # Format chat history
        history_text = ""
        if chat_history:
            for turn in chat_history[-5:]:  # Include only last 5 turns to save context
                history_text += f"User: {turn['user']}\n"
                history_text += f"Thallan: {turn['character']}\n\n"

        # Build system prompt with stronger anti-hallucination constraints
        system_prompt = f"""You are Thallan, a {self.persona['race']} {self.persona['occupation']} living in the Radiant Citadel. You must follow these ABSOLUTE RULES:

1. ONLY respond with information found in the KNOWLEDGE CONTEXT section below.
2. NEVER invent, assume, or create facts not explicitly stated in the KNOWLEDGE CONTEXT.
3. If you don't know something, you MUST say: "That is not something I know, traveler. Perhaps a more seasoned scholar or explorer might."
4. Analyze the user's question carefully to determine if you have sufficient context to answer it properly.
5. If there's insufficient context for a specific question, ADMIT you don't know rather than making up an answer.
6. You must answer as a person within the world, NEVER reference any information as if this is a game that is being played with a DM and players.

CHARACTER INFORMATION:
- Background: {self.persona.get('background', 'Sage')}
- Personality traits:{personality}
- Speech patterns:{speech}
- Knowledge specialties:{knowledge}

RESPONSE REQUIREMENTS:
- All responses must be in Thallan's voice, using the speech patterns described above.
- Never break character or explain your behavior as an AI or reference this being a simulation.
- Only answer from the perspective of your character's knowledge and experience.
- Thallan should be helpful, warm, and eager to share knowledge about the Radiant Citadel.

KNOWLEDGE CONTEXT (THIS IS THE ONLY INFORMATION YOU CAN USE):
{lore_context}

CONVERSATION HISTORY:
{history_text}"""

        # Simple format that works well with most open-source LLMs
        prompt = f"{system_prompt}\n\nUser: {user_input}\n\nThallan:"

        return {
            "prompt": prompt,
            "context": lore_context
        }

    def _get_uncertainty_response(self) -> str:
        """
        Generate a response for when Thallan doesn't know the answer.
        Adds variety to uncertainty responses.

        Returns:
            A response indicating lack of knowledge
        """
        import random

        uncertainty_responses = [
            "That is not something I know, traveler. Perhaps a more seasoned scholar or explorer might have that information.",
            "I'm afraid the records of the Radiant Citadel do not contain that information—at least not in the sections I've studied.",
            "In my studies of the Citadel's histories, I haven't encountered details about that. The Dawn Incarnates might know more.",
            "Ah, an interesting question indeed, but one that lies beyond my knowledge. The Amaranthine Market has many wise travelers who might help you.",
            "The Concord does not reveal all things to all people. I regret that I cannot provide you with that particular wisdom.",
            "While I pride myself on my knowledge of the Radiant Citadel, I must admit that this topic falls outside my area of expertise."
        ]

        return random.choice(uncertainty_responses)

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

        # Build the prompt with improved anti-hallucination measures
        prompt_data = self._build_prompt(user_input, chat_history)
        prompt = prompt_data["prompt"]
        context = prompt_data["context"]

        # Log GPU info if available and verbose mode is on
        if torch.cuda.is_available() and self.verbose:
            print(f"GPU memory before generation: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # Generate response
        try:
            # Use a lower temperature for more factual responses
            adjusted_temp = min(temperature, 0.5) if self.strict_knowledge else temperature

            response = self.llm(
                prompt,
                max_new_tokens=max_tokens,
                temperature=adjusted_temp,
                do_sample=True,
                top_p=0.85,  # More conservative sampling
                repetition_penalty=1.2,  # Discourage repetition
                num_return_sequences=1
            )

            # Log GPU info if available and verbose mode is on
            if torch.cuda.is_available() and self.verbose:
                print(f"GPU memory after generation: {torch.cuda.memory_allocated()/1e9:.2f} GB")

            # Extract generated text
            generated_text = response[0]['generated_text'].strip()

            # Clean up any spurious tags or markers
            end_markers = ["User:", "USER:", "Player:", "<|endoftext|>", "</s>"]
            for marker in end_markers:
                if marker in generated_text:
                    generated_text = generated_text.split(marker)[0].strip()

            # Hallucination check - only if strict knowledge mode is on
            if self.strict_knowledge:
                # If response contains hallucinated information, replace with uncertainty response
                if not self._is_response_grounded(generated_text, context):
                    if self.verbose:
                        print("Detected potential hallucination - replacing response")
                    return self._get_uncertainty_response()

            return generated_text

        except Exception as e:
            if self.verbose:
                print(f"Error during generation: {e}")
            return "I apologize, traveler. My thoughts seem muddled at the moment. Could you perhaps phrase your question differently?"


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