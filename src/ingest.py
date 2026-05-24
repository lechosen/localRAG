from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from src.config import CHROMA_DIR, build_embeddings

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 50  # number of chunks sent to Ollama per embedding call


def load_documents(source: str):
    # based on source, support URL, PDF and *.txt
    if source.startswith("http://") or source.startswith("https://"):
        loader = WebBaseLoader(source)
    elif source.lower().endswith(".pdf"):
        loader = PyPDFLoader(source)
    else:
        loader = TextLoader(source, encoding="utf-8")
    return loader.load()


def ingest(source: str, display_name: str | None = None) -> int:
    # display_name is the human-readable label stored in metadata["source"].
    # For file uploads, pass the original filename so source attribution is
    # meaningful and stable across re-uploads (temp paths change every time).
    name = display_name or os.path.basename(source)

    docs = load_documents(source)

    # Override source metadata so all chunks carry the clean display name,
    # not the ephemeral Gradio temp path.
    for doc in docs:
        doc.metadata["source"] = name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    embeddings = build_embeddings()
    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

    # Remove any previously ingested chunks for this source so re-ingesting
    # replaces rather than duplicates. Without this, every re-upload appends
    # a full copy to the collection.
    existing = db.get(where={"source": name})
    if existing["ids"]:
        db.delete(ids=existing["ids"])

    # Embed and insert in batches to avoid sending hundreds of chunks to
    # Ollama in a single HTTP request, which can stall or timeout for large PDFs.
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        db.add_documents(chunks[i : i + EMBED_BATCH_SIZE])

    return len(chunks)
