"""
Case Q&A — RAG chat grounded in course / case materials.

Answers questions using ONLY uploaded documents, with citations.
Supports uploading PDFs directly from the browser (no terminal needed).
Refuses to answer when the indexed documents don't support the question.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import streamlit as st

COLLECTION_NAME = "case_materials"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
DISTANCE_THRESHOLD = 1.3   # cosine distance 0-2; 1.3 is permissive enough for relevant content
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

NOT_FOUND_MSG = (
    "Not enough information in the provided materials to answer this question. "
    "Please consult your course materials directly."
)

SYSTEM_PROMPT = (
    "You are a teaching assistant. Answer the question using ONLY the context "
    "provided below. Do not use any prior knowledge or information outside the "
    "context. If the context does not sufficiently support an answer, respond "
    f'with exactly: "{NOT_FOUND_MSG}"'
)


# ────────────────────────  ChromaDB client  ────────────────────────

def _get_chroma_client():
    """Return an ephemeral ChromaDB client stored in session state.

    Using EphemeralClient (in-memory) avoids all disk-path and SQLite-version
    issues on Streamlit Cloud.  The client survives Streamlit reruns because
    it lives in st.session_state, which persists for the lifetime of the
    browser session.
    """
    if "cqa_chroma_client" not in st.session_state:
        import chromadb
        st.session_state.cqa_chroma_client = chromadb.EphemeralClient()
    return st.session_state.cqa_chroma_client


def _get_or_create_collection():
    """Get (or create) the ChromaDB collection from the session-state client."""
    client = _get_chroma_client()
    try:
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


def _get_collection():
    """Return the collection if it has chunks, else None."""
    try:
        col = _get_or_create_collection()
        return col if col.count() > 0 else None
    except Exception:
        return None


# ────────────────────────  chunking  ────────────────────────

def chunk_text(text: str, source: str, page: int = 0) -> list[dict]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    chunks: list[dict] = []
    i = 0
    while i < len(words):
        chunk_words = words[i: i + CHUNK_SIZE]
        chunk_text_str = " ".join(chunk_words)
        chunk_id = f"{source}_p{page}_c{i}"
        chunks.append({
            "text": chunk_text_str,
            "source": source,
            "page": page,
            "chunk_id": chunk_id,
        })
        if len(chunk_words) < CHUNK_SIZE:
            break
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ────────────────────────  indexing  ────────────────────────

def _build_index_from_bytes(file_bytes: bytes, filename: str, openai_client) -> int:
    """Extract text, chunk, embed, and store in ChromaDB. Returns chunk count added."""
    all_chunks: list[dict] = []
    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_chunks.extend(chunk_text(page_text, filename, page_num))
        except Exception as e:
            st.error(f"Failed to read PDF '{filename}': {e}")
            return 0
    elif filename.lower().endswith((".md", ".txt")):
        text = file_bytes.decode("utf-8", errors="replace")
        all_chunks.extend(chunk_text(text, filename, 0))
    else:
        st.warning(f"Unsupported file type: '{filename}'. Only PDF, MD, and TXT are supported.")
        return 0

    if not all_chunks:
        st.warning(
            f"No text could be extracted from '{filename}'. "
            "If this is a scanned PDF, text extraction is not supported — "
            "please use a PDF with selectable text."
        )
        return 0

    collection = _get_or_create_collection()

    batch_size = 100
    added = 0
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]
        try:
            resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
            embeddings = [r.embedding for r in resp.data]
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            return added

        ids = [c["chunk_id"] for c in batch]
        metas = [{"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]} for c in batch]

        # Skip duplicate IDs
        existing = set(collection.get(ids=ids)["ids"])
        new_idx = [j for j, cid in enumerate(ids) if cid not in existing]
        if new_idx:
            collection.add(
                ids=[ids[j] for j in new_idx],
                documents=[texts[j] for j in new_idx],
                embeddings=[embeddings[j] for j in new_idx],
                metadatas=[metas[j] for j in new_idx],
            )
            added += len(new_idx)

    return added


# ────────────────────────  retrieval  ────────────────────────

def _embed(text: str, openai_client) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _retrieve(question: str, collection, openai_client) -> tuple[list[str], list[str]]:
    """Return (relevant_doc_texts, citation_strings)."""
    query_vec = _embed(question, openai_client)
    n = min(TOP_K, collection.count())
    if n == 0:
        return [], []

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    relevant_docs, citations = [], []
    for doc, meta, dist in zip(docs, metas, dists):
        if dist <= DISTANCE_THRESHOLD:
            relevant_docs.append(doc)
            source = meta.get("source", "unknown")
            page = meta.get("page", "?")
            chunk_id = meta.get("chunk_id", "?")
            citations.append(f"`{source}` — page {page} (chunk `{chunk_id}`)")

    return relevant_docs, citations


def _answer(question: str, context_docs: list[str], openai_client) -> str:
    if not context_docs:
        return NOT_FOUND_MSG
    context = "\n\n---\n\n".join(context_docs)
    completion = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    return completion.choices[0].message.content


# ────────────────────────  UI  ────────────────────────

def show_case_qa() -> None:
    # Back button
    col_back, _ = st.columns([1, 11])
    with col_back:
        if st.button("← Back", key="back_caseqa"):
            st.session_state.current_view = "home"
            st.rerun()

    st.markdown("""
<div style="padding: 10px 0 10px 0;">
    <h2 style="color: #e6edf3;">📚 Case Q&amp;A</h2>
    <p style="color: #8b949e;">
        Ask questions about your course materials.
        Answers are grounded <em>only</em> in documents you upload here — with file &amp; page citations.
        Unsupported questions are refused explicitly.
    </p>
</div>
""", unsafe_allow_html=True)

    # ── API KEY ───────────────────────────────────────────────────────────────
    api_key = ""
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        st.error("**OPENAI_API_KEY** not set. Add it to your `.env` file or Streamlit secrets.")
        return

    import openai
    openai_client = openai.OpenAI(api_key=api_key)

    # ── FILE UPLOADER ─────────────────────────────────────────────────────────
    st.write("### Upload Course Materials")
    st.caption(
        "Upload PDFs, Markdown, or text files. "
        "They will be indexed automatically — no terminal commands needed. "
        "**Note:** documents are held in memory for this session; re-upload if you refresh the page."
    )

    uploaded_files = st.file_uploader(
        "Drop files here or click to browse",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        key="caseqa_uploader",
    )

    if uploaded_files:
        files_to_index = []
        for uf in uploaded_files:
            indexed_key = f"cqa_indexed_{uf.name}_{uf.size}"
            if not st.session_state.get(indexed_key):
                files_to_index.append(uf)

        if files_to_index:
            with st.spinner(f"Indexing {len(files_to_index)} file(s)… this may take a moment."):
                total_added = 0
                for uf in files_to_index:
                    file_bytes = uf.read()
                    n = _build_index_from_bytes(file_bytes, uf.name, openai_client)
                    total_added += n
                    st.session_state[f"cqa_indexed_{uf.name}_{uf.size}"] = True

            if total_added > 0:
                st.success(f"Indexed {total_added} chunk(s) from {len(files_to_index)} file(s). Ready to chat!")
            else:
                st.info("No new chunks were added (files may already be indexed or contained no extractable text).")
        else:
            st.info(f"{len(uploaded_files)} file(s) already indexed this session.")

    # ── CHAT INTERFACE ────────────────────────────────────────────────────────
    collection = _get_collection()
    if collection is None or collection.count() == 0:
        st.divider()
        st.warning(
            "**No documents indexed yet.**\n\n"
            "Upload PDFs or text files above to get started."
        )
        return

    st.divider()
    st.caption(f"Index ready — {collection.count()} chunks from your uploaded documents.")

    # Render chat history
    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    for msg in st.session_state.qa_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # New question
    question = st.chat_input("Ask a question about the case materials…")
    if question:
        st.session_state.qa_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                docs, citations = _retrieve(question, collection, openai_client)
                answer = _answer(question, docs, openai_client)

            response = answer
            if citations and NOT_FOUND_MSG not in answer:
                response += "\n\n**Sources:**\n" + "\n".join(f"- {c}" for c in citations)

            st.markdown(response)

        st.session_state.qa_messages.append({"role": "assistant", "content": response})
