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
   Ensure you have a `.env` file in the root directory and add your keys (e.g., `OPENROUTER_API_KEY`).

3. **Run the FastAPI Server:**
   ```bash
   python API/main.py
   ```
   The API will be available at `http://localhost:8000`.

### Frontend Setup

The frontend is built with vanilla web technologies, so no Node/NPM build step is required! 

To view the interface, start a simple HTTP server in the `frontend` directory:

```bash
cd frontend
python -m http.server 8001
```

Navigate to `http://localhost:8001` in your browser. The UI will automatically connect to your running FastAPI backend. Look for the green dot in the sidebar to confirm the system is online!

## API Endpoints

- **`GET /health`** or **`GET /api/v1/rag/health`**: Application health check.
- **`POST /api/v1/rag/get_answer`**: Submits a query and retrieves a RAG-augmented response.
  - **Payload Example**: `{"query": "What is the attention mechanism?", "top_k": 5}`

## Contributing
Feel free to submit issues and enhancement requests.
