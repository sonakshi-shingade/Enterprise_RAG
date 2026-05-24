# Enterprise RAG System

A production-grade Retrieval-Augmented Generation (RAG) system built with FastAPI, ChromaDB, Sentence Transformers, and OpenRouter, accompanied by a modern Vanilla HTML/CSS/JS chat interface.

## Architecture

This project is separated into a robust backend API and a lightweight frontend interface.

### Backend Stack
- **Framework**: FastAPI (Python)
- **Vector Database**: ChromaDB
- **Embeddings**: SentenceTransformers (`paraphrase-MiniLM-L6-v2`)
- **Semantic Caching**: ChromaDB answer cache for repeated or semantically similar queries
- **LLM Provider**: OpenRouter (`meta-llama/llama-3.3-70b-instruct`)
- **Telegram Integration**: Native Telegram bot support via `aiogram` for direct RAG query access from chat.

### Frontend Stack
- **Technologies**: Vanilla HTML, CSS, JavaScript
- **Features**: Glassmorphism aesthetic, responsive design, configurable 'Top K' retrieval settings, live backend health monitoring.

## Directory Structure

```text
RAG/
├── API/                 # FastAPI application and routing logic
├── Database/            # Persistent storage for ChromaDB
├── Documents/           # Raw documents for ingestion
├── Embedding_Model/     # Local storage for embedding model weights
├── Guardrails/          # Output validation logic
├── Services/            # Core RAG services (ChromaDB, LLM, Embeddings, Ingestion)
├── Utils/               # Utility scripts and the main RAG pipeline orchestrator
├── frontend/            # Vanilla HTML/JS/CSS Chat User Interface
├── .env                 # Environment variables configuration
└── .gitignore           # Git ignore rules
```

## Getting Started

### Prerequisites

- Python 3.8+
- An OpenRouter API Key (placed in your `.env` file)

### Backend Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Configure Environment Variables:**
   Ensure you have a `.env` file in the root directory and add your keys, for example:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   ```
   If `TELEGRAM_BOT_TOKEN` is set, the Telegram bot will start automatically with the backend.

3. **Run the FastAPI Server:**
   ```bash
   python API/main.py
   ```
   The API will be available at `http://localhost:8200`.

### Frontend Setup

The frontend is built with vanilla web technologies, so no Node/NPM build step is required! 

To view the interface, start a simple HTTP server in the `frontend` directory:

```bash
cd frontend
python -m http.server 8001
```

Navigate to `http://localhost:8001` in your browser. The UI will automatically connect to your running FastAPI backend. Look for the green dot in the sidebar to confirm the system is online!

## Telegram Integration

The RAG backend can also serve queries directly from Telegram. When `TELEGRAM_BOT_TOKEN` is configured in `.env`, the app starts an `aiogram` bot that listens for text messages and forwards them through the same RAG pipeline.

- Send `/start` to begin a new chat session.
- Type any question to receive a RAG-augmented answer in Telegram.
- The bot runs in the background while the FastAPI server is running.

## API Endpoints

- **`GET /health`** or **`GET /api/v1/rag/health`**: Application health check.
- **`POST /api/v1/rag/get_answer`**: Submits a query and retrieves a RAG-augmented response.
  - **Payload Example**: `{"query": "What is the attention mechanism?", "top_k": 5}`

## Contributing
Feel free to submit issues and enhancement requests.
