# LocalRAG Chatbot

A fully local Retrieval-Augmented Generation (RAG) Q&A system. No cloud API required — all models run on your machine via Ollama.

## Architecture

```
User Question (Gradio UI)
        │
        ▼
    app.py  ── Gradio frontend: handles file uploads, URLs, and questions
        ├── ingest.py     ── Load → chunk → embed (batched) → deduplicate → store in ChromaDB
        └── rag_chain.py  ── Retrieve → fill prompt → LLM → return answer + sources
                                    │
                                Ollama (localhost:11434)
                                    ├── mxbai-embed-large  (embeddings)
                                    └── gemma3:4b          (answer generation)
```

### Modules

| File | Responsibility |
|------|---------------|
| `src/config.py` | Reads all model names and paths from `.env`; provides `build_llm()` and `build_embeddings()` via a provider registry |
| `src/ingest.py` | Loads PDF / TXT / URL, splits into chunks (500 tokens, 50 overlap), deduplicates by source name, and writes to ChromaDB in batches of 50 |
| `src/rag_chain.py` | Retrieves top-5 relevant chunks, injects them into a strict prompt template, calls the LLM, and returns the answer with source attribution |
| `src/app.py` | Gradio web UI with two tabs: Document Ingestion and Q&A |

### Supported Providers

`config.py` uses a registry pattern — switching providers only requires changing `.env`, no code changes needed.

| Type | Supported Providers |
|------|-------------------|
| LLM | `ollama`, `openai`, `anthropic`, `deepseek` |
| Embeddings | `ollama`, `openai` |

> **Note:** Switching `EMBED_MODEL` requires re-running `ingest.py` to rebuild the vector store. Different embedding models are incompatible with each other.

## Requirements

- [Ollama](https://ollama.com) installed and running
- Python 3.11+

## Setup

**1. Pull models**
```bash
ollama pull gemma3:4b
ollama pull mxbai-embed-large
```

**2. Create virtual environment and install dependencies**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Do **not** install `unstructured` — it pulls in `llvmlite` which fails to compile on most setups.

**3. Configure environment**

Copy `.env.example` to `.env` and adjust as needed:
```
LLM_PROVIDER=ollama
LLM_MODEL=gemma3:4b
EMBED_PROVIDER=ollama
EMBED_MODEL=mxbai-embed-large
CHROMA_DIR=./chroma_db/mxbai-embed-large
```

## Running

```bash
# Terminal A — start Ollama
ollama serve

# Terminal B — start the app
source .venv/bin/activate
python main.py
```

Open the URL shown in the terminal (typically `http://localhost:7860`) in your browser.

**Ingest a document**, then switch to the **Q&A** tab to ask questions about it.

## Improvement

This project uses a basic RAG pipeline (fixed chunking → vector indexing → similarity retrieval → generation). Two reference documents outline potential improvements:

- [rag-17-strategies.md](rag-17-strategies.md) — 17 optimization strategies across chunking, indexing, retrieval, and advanced techniques (e.g. semantic chunking, HyDE, re-ranking, Fusion RAG, CRAG). Each strategy includes a code reference and the core library to implement it.

- [rag-evaluation.md](rag-evaluation.md) — How to measure RAG quality across three dimensions: answer accuracy, LLM utilization, and retrieval effectiveness (precision, recall, MRR, NDCG). Includes tooling recommendations such as RAGAS and DeepEval.