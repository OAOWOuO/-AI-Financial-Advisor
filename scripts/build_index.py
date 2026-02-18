"""
Build the RAG index from documents in data/raw/.

Usage:
    python scripts/build_index.py

This script will:
  1. Load all .pdf, .md, and .txt files from data/raw/
  2. Chunk each document (500 words, 50-word overlap)
  3. Save chunks to data/processed/chunks.json  (reproducibility)
  4. Embed every chunk with OpenAI text-embedding-3-small
  5. Store vectors + metadata in ChromaDB at index/

Requirements:
  pip install openai chromadb pypdf python-dotenv
  OPENAI_API_KEY must be set in .env
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
INDEX_DIR = Path("index")
CHUNK_SIZE = 500        # words per chunk
OVERLAP = 50            # word overlap between adjacent chunks
EMBED_MODEL = "text-embedding-3-small"
COLLECTION_NAME = "case_materials"
BATCH_SIZE = 100        # embeddings per API call


def chunk_text(text: str, source: str, page: int) -> list[dict]:
    """Split *text* into overlapping word-window chunks."""
    words = text.split()
    chunks: list[dict] = []
    step = CHUNK_SIZE - OVERLAP
    for i, start in enumerate(range(0, len(words), step)):
        body = " ".join(words[start: start + CHUNK_SIZE])
        if body.strip():
            chunks.append({
                "text": body,
                "source": source,
                "page": page,
                "chunk_id": f"{source}_p{page}_c{i}",
            })
    return chunks


def load_documents() -> list[dict]:
    """Load text from all supported files in RAW_DIR."""
    docs: list[dict] = []

    # --- PDFs ---
    try:
        from pypdf import PdfReader
        for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
            print(f"  Loading PDF : {pdf_path.name}")
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({"text": text, "source": pdf_path.name, "page": page_num + 1})
    except ImportError:
        print("  [warn] pypdf not installed — skipping PDFs.  Run: pip install pypdf")

    # --- Markdown / plain-text files ---
    for ext in ("*.md", "*.txt"):
        for path in sorted(RAW_DIR.glob(ext)):
            if path.name == "README.md":
                continue
            print(f"  Loading text: {path.name}")
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append({"text": text, "source": path.name, "page": 1})

    return docs


def embed_texts(texts: list[str], client) -> list[list[float]]:
    """Embed *texts* in batches, returning one vector per text."""
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        embeddings.extend(e.embedding for e in resp.data)
        print(f"  Embedded {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")
    return embeddings


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("Step 1/4  Loading documents from data/raw/ ...")
    docs = load_documents()
    if not docs:
        print("No documents found.  Add PDFs or .md files to data/raw/ and re-run.")
        sys.exit(0)
    print(f"  Loaded {len(docs)} page(s) / document(s).")

    # 2. Chunk
    print("Step 2/4  Chunking text ...")
    all_chunks: list[dict] = []
    for doc in docs:
        all_chunks.extend(chunk_text(doc["text"], doc["source"], doc["page"]))
    print(f"  {len(all_chunks)} chunks created.")

    # 3. Persist chunks (reproducibility artefact)
    chunks_path = PROCESSED_DIR / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    print(f"  Chunks saved to {chunks_path}")

    # 4. Embed
    print("Step 3/4  Embedding chunks with OpenAI ...")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set.  Add it to your .env file.")
        sys.exit(1)
    import openai
    client = openai.OpenAI(api_key=api_key)
    embeddings = embed_texts([c["text"] for c in all_chunks], client)

    # 5. Store in ChromaDB
    print("Step 4/4  Writing to ChromaDB vector store ...")
    import chromadb
    chroma = chromadb.PersistentClient(path=str(INDEX_DIR))
    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=[c["chunk_id"] for c in all_chunks],
        embeddings=embeddings,
        documents=[c["text"] for c in all_chunks],
        metadatas=[
            {"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]}
            for c in all_chunks
        ],
    )
    print(f"\nDone!  Index built with {len(all_chunks)} chunks stored at '{INDEX_DIR}/'.")
    print("Start the app:  streamlit run streamlit_app/app.py")


if __name__ == "__main__":
    main()
