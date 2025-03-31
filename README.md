# Thallan: D&D NPC AI Assistant

![image](https://github.com/user-attachments/assets/4c1443c1-aa75-4a20-9acd-d2665d682a33)

## Project Overview

Thallan is an AI-powered D&D NPC (Non-Player Character) from the Radiant Citadel. This project creates a conversational character that:

- Responds to questions about D&D lore from the Radiant Citadel sourcebook
- Maintains a consistent character persona and speech patterns
- Provides useful guidance to players during a D&D campaign

This project demonstrates several key AI and NLP techniques:
- **Retrieval-Augmented Generation (RAG)** - Enhances AI responses with relevant D&D lore
- **Vector embeddings** for powerful semantic search capabilities
- **Local LLM inference** optimized for both CPU and GPU environments
- **Character personality modeling** with consistent persona maintenance

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 🎮 Demo

![image](https://github.com/user-attachments/assets/0f086fc4-960b-443d-9cca-e6018fc93eef)


## 🏗️ Technical Architecture

### Components

1. **Data Processing Pipeline**
   - Parses D&D sourcebook materials from markdown format
   - Chunks text into appropriate sizes for embeddings
   - Preserves metadata for better context retrieval

2. **Vector Database**
   - Uses sentence-transformers for creating embeddings
   - Stores vectors in FAISS for efficient similarity search
   - Retrieves relevant lore based on semantic similarity to player queries

3. **Character Persona**
   - Defines Thallan's personality, background, and speech patterns
   - Templates prompts for consistent character responses
   - Combines retrieved lore with character voice

4. **User Interfaces**
   - CLI interface with rich text formatting
   - Simple web interface using FastAPI and WebSockets

### Technologies Used

- **Python**: Core language
- **LangChain**: Framework for LLM applications
- **Sentence-Transformers**: Fast CPU-friendly embeddings
- **FastAPI**: Web interface backend
- **Typer/Rich**: CLI interface
- **HuggingFace Transformers**: Model loading and inference

## 🚀 Installation & Setup

### Prerequisites

- Python 3.9+
- 2GB+ RAM for embeddings
- 4GB+ RAM for LLM inference
- (Optional) CUDA-compatible GPU for faster inference

### Step-by-Step Installation

1. **Clone the repository**:
```bash
git clone https://github.com/[YOUR_USERNAME]/thallan-radiant-citadel.git
cd thallan-radiant-citadel
```

2. **Create a virtual environment**:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt

# Optional: For GPU acceleration (if you have NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

4. **Check GPU availability** (optional):
```bash
python -m src.gpu_check
```

5. **Download a model**:
```bash
# For CPU usage
python download_model.py --model microsoft/phi-2

# To see all model options
python download_model.py --list-models
```

6. **Prepare your data**:
   - Place your Radiant Citadel markdown files in `data/raw/`
   - Run the processing pipeline: `python -m app.cli process`
   - Create embeddings: `python -m app.cli embed`

7. **Run the complete setup with a single command**:
```bash
python -m app.cli setup
```

## 💬 Usage

### CLI Interface

Start the CLI to chat with Thallan:
```bash
# Basic usage with default model
python -m app.cli chat

# With GPU acceleration (if available)
python -m app.cli chat --force-gpu

# Using a specific model
python -m app.cli chat --model-path microsoft/phi-2 --force-gpu
```

View available models:
```bash
python -m app.cli models
```

Check your GPU configuration:
```bash
python -m app.cli gpu-info
```

### Web Interface

Start the web server:
```bash
python -m app.web
```

Then open your browser to `http://localhost:8000`

### Using in Your Campaign

1. **Prep Phase**: Run Thallan before your session to prepare for likely player questions
2. **During Play**: Use Thallan when players interact with the character
3. **Between Sessions**: Review chat history to inform campaign development

## 🧪 Project Structure

```
thallan-radiant-citadel/
├── app/                    # Application interfaces
│   ├── cli.py              # Command-line interface
│   ├── web.py              # Web interface with FastAPI
│   └── templates/          # Web templates (auto-generated)
├── data/                   # Data directory
│   ├── raw/                # Raw markdown files
│   └── processed/          # Processed chunks and embeddings
├── src/                    # Core source code
│   ├── character.py        # Character persona and generation
│   ├── data_processing.py  # Text processing pipeline
│   ├── embeddings.py       # Vector embedding management
│   ├── gpu_check.py        # GPU diagnostics
│   ├── retrieval.py        # Semantic search & retrieval
│   └── utils.py            # Utility functions
├── download_model.py       # Model download utility
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

## 💼 Portfolio Value

This project showcases various technical skills valuable for machine learning and AI engineering roles:

### Software Engineering
- Proper packaging and project organization
- Command-line and web interfaces
- GPU optimization and management

### Machine Learning & NLP
- Text embeddings and vector search
- Prompt engineering
- LLM optimization for resource constraints
- Retrieval-Augmented Generation (RAG)

### Data Engineering
- Text processing pipeline
- Metadata extraction and management
- Vector database implementation

## 🔮 Future Enhancements

- Support for multiple NPCs with different personas
- Advanced GPU memory optimization
- Docker container for simplified deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/some-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/some-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The D&D Radiant Citadel source material is owned by Wizards of the Coast
- This project is for educational purposes only
