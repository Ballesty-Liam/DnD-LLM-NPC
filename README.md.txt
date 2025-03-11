# Thallan: D&D NPC AI Assistant

![Radiant Citadel](https://via.placeholder.com/800x200?text=The+Radiant+Citadel)

## Project Overview

Thallan is an AI-powered D&D NPC (Non-Player Character) from the Radiant Citadel. This project creates a conversational character that:

- Responds to questions about D&D lore from the Radiant Citadel sourcebook
- Maintains a consistent character persona and speech patterns
- Provides useful guidance to players during a D&D campaign

This project demonstrates several key AI and NLP techniques:
- Retrieval-Augmented Generation (RAG)
- Vector embeddings for semantic search
- Local LLM inference without GPU requirements
- Character personality modeling

## Technical Architecture

### Components

1. **Data Processing Pipeline**
   - Parses D&D sourcebook materials from markdown format
   - Chunks text into appropriate sizes for embeddings
   - Preserves metadata for better context retrieval

2. **Vector Database**
   - Uses sentence-transformers for creating embeddings
   - Stores vectors in ChromaDB for efficient similarity search
   - Retrieves relevant lore based on semantic similarity to player queries

3. **Character Persona**
   - Defines Thallan's personality, background, and speech patterns
   - Templates prompts for consistent character responses
   - Combines retrieved lore with character voice

4. **Inference**
   - Uses llama-cpp-python for optimized CPU inference
   - Supports multiple lightweight LLMs

5. **User Interfaces**
   - CLI interface with rich text formatting
   - Simple web interface using FastAPI and WebSockets

### Technologies Used

- **Python**: Core language
- **LangChain**: Framework for LLM applications
- **Sentence-Transformers**: Fast CPU-friendly embeddings
- **ChromaDB**: Vector database
- **Llama-cpp-python**: Optimized LLM inference
- **FastAPI**: Web interface backend
- **Typer/Rich**: CLI interface

## Installation & Setup

### Prerequisites

- Python 3.9+
- 2GB+ RAM for embeddings
- 4GB+ RAM for LLM inference

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/radiant_citadel_npc.git
cd radiant_citadel_npc
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download a model:
```bash
python scripts/download_model.py
```

5. Prepare your data:
   - Place your Radiant Citadel markdown files in `data/raw/`
   - Run the processing pipeline: `python -m app.cli process`
   - Create embeddings: `python -m app.cli embed`

### Usage

#### CLI Interface

Start the CLI to chat with Thallan:
```bash
python -m app.cli chat
```

Or run the full setup process:
```bash
python -m app.cli setup
```

#### Web Interface

Start the web server:
```bash
python -m app.web
```

Then open your browser to `http://localhost:8000`

## For D&D Dungeon Masters

### Customizing Thallan

You can customize Thallan's personality by editing the persona data in `src/character.py` or by creating a custom JSON file:

```json
{
  "name": "Thallan",
  "race": "Half-Elf",
  "occupation": "Scholar and guide at the Radiant Citadel",
  "background": "Sage",
  "personality": [
    "Knowledgeable and scholarly",
    "Warm and welcoming to visitors"
  ],
  "speech_patterns": [
    "Uses educational metaphors",
    "Occasionally references obscure Citadel lore"
  ],
  "knowledge_specialties": [
    "Radiant Citadel history and layout",
    "The Founders and Dawn Incarnates"
  ]
}
```

### Using in Your Campaign

1. **Prep Phase**: Run Thallan before your session to prepare for likely player questions
2. **During Play**: Use Thallan when players interact with the character
3. **Between Sessions**: Review chat history to inform campaign development

## Portfolio Value

This project showcases various technical skills valuable for machine learning and AI engineering roles:

### Software Engineering
- Clean, modular code structure 
- Proper packaging and project organization
- Command-line and web interfaces
- Comprehensive documentation

### Machine Learning & NLP
- Text embeddings and vector search
- Prompt engineering
- LLM optimization for resource constraints
- Retrieval-Augmented Generation

### Data Engineering
- Text processing pipeline
- Metadata extraction and management
- Vector database implementation

### UI/UX
- Interactive CLI with rich formatting
- Responsive web interface with WebSockets

## Future Enhancements

- Support for multiple NPCs with different personas
- Voice output using TTS
- Integration with VTT platforms like Roll20 or Foundry
- Fine-tuning capabilities for custom knowledge domains

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The D&D Radiant Citadel source material is owned by Wizards of the Coast
- This project is for educational purposes only