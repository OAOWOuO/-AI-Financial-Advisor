"""
Case Q&A — RAG chat grounded in course / case materials.

Answers questions using ONLY uploaded documents, with citations.
Supports uploading PDFs directly from the browser (no terminal needed).
Refuses to answer when the indexed documents don't support the question.
"""
from __future__ import annotations

import io
import os

import streamlit as st

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 8          # more chunks = better coverage for broad / summary questions
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

NOT_FOUND_MSG = (
    "Not enough information in the provided materials to answer this question. "
    "Please consult your course materials directly."
)

SYSTEM_PROMPT = f"""You are a teaching assistant. Students ask questions about their uploaded course materials.

You are given excerpts from those materials below. Use them to answer the student's question.

Rules:
1. Base your answer ONLY on the provided excerpts — do not use outside knowledge.
2. For general questions such as "summarize", "what is this about", "explain this document", \
"tell me about the file", or similar — synthesize an answer from the excerpts even if no single \
excerpt says "this document is about X".
3. Answer specific questions about facts, concepts, or topics if the excerpts cover them.
4. ONLY if the excerpts are completely unrelated to the question (e.g. the student asks about \
the weather or a topic not mentioned anywhere in the excerpts), respond with exactly:
"{NOT_FOUND_MSG}"
"""


# ────────────────────────  In-memory store (no ChromaDB)  ────────────────────────

def _get_store() -> dict:
    if "cqa_store" not in st.session_state:
        st.session_state.cqa_store = {
            "ids": [],
            "documents": [],
            "embeddings": [],
            "metadatas": [],
        }
    return st.session_state.cqa_store


def _store_count() -> int:
    return len(_get_store()["ids"])


def _add_chunks(ids: list, documents: list, embeddings: list, metadatas: list) -> int:
    store = _get_store()
    existing = set(store["ids"])
    added = 0
    for i, id_ in enumerate(ids):
        if id_ not in existing:
            store["ids"].append(id_)
            store["documents"].append(documents[i])
            store["embeddings"].append(embeddings[i])
            store["metadatas"].append(metadatas[i])
            existing.add(id_)
            added += 1
    return added


def _top_k(query_embedding: list[float], n: int) -> tuple[list[str], list[dict]]:
    import numpy as np
    store = _get_store()
    if not store["embeddings"]:
        return [], []

    q = np.array(query_embedding, dtype="float32")
    q /= np.linalg.norm(q) + 1e-10

    scores = []
    for i, emb in enumerate(store["embeddings"]):
        e = np.array(emb, dtype="float32")
        e /= np.linalg.norm(e) + 1e-10
        scores.append(float(q @ e))

    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
    docs = [store["documents"][i] for i in top_idx]
    metas = [store["metadatas"][i] for i in top_idx]
    return docs, metas


# ────────────────────────  chunking  ────────────────────────

def _chunk_text(text: str, source: str, page: int = 0) -> list[dict]:
    words = text.split()
    chunks: list[dict] = []
    i = 0
    while i < len(words):
        window = words[i: i + CHUNK_SIZE]
        chunks.append({
            "text": " ".join(window),
            "source": source,
            "page": page,
            "chunk_id": f"{source}_p{page}_c{i}",
        })
        if len(window) < CHUNK_SIZE:
            break
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ────────────────────────  indexing  ────────────────────────

def _build_index_from_bytes(file_bytes: bytes, filename: str, openai_client) -> int:
    all_chunks: list[dict] = []

    if filename.lower().endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    all_chunks.extend(_chunk_text(page_text, filename, page_num))
        except Exception as e:
            st.error(f"Failed to read PDF '{filename}': {e}")
            return 0
    elif filename.lower().endswith((".md", ".txt")):
        text = file_bytes.decode("utf-8", errors="replace")
        all_chunks.extend(_chunk_text(text, filename, 0))
    else:
        st.warning(f"Unsupported file type: '{filename}'. Only PDF, MD, and TXT are supported.")
        return 0

    if not all_chunks:
        st.warning(
            f"No text could be extracted from '{filename}'. "
            "If this is a scanned PDF, text extraction is not supported — "
            "please use a PDF with selectable (digital) text."
        )
        return 0

    existing_ids = set(_get_store()["ids"])
    new_chunks = [c for c in all_chunks if c["chunk_id"] not in existing_ids]
    if not new_chunks:
        return 0

    total_added = 0
    batch_size = 100
    for i in range(0, len(new_chunks), batch_size):
        batch = new_chunks[i: i + batch_size]
        texts = [c["text"] for c in batch]
        try:
            resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
            embeddings = [r.embedding for r in resp.data]
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            return total_added

        ids = [c["chunk_id"] for c in batch]
        metas = [{"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]} for c in batch]
        total_added += _add_chunks(ids, texts, embeddings, metas)

    return total_added


# ────────────────────────  retrieval  ────────────────────────

def _embed(text: str, openai_client) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _retrieve(question: str, openai_client) -> tuple[list[str], list[str]]:
    query_vec = _embed(question, openai_client)
    n = min(TOP_K, _store_count())
    if n == 0:
        return [], []

    docs, metas = _top_k(query_vec, n)
    citations = [
        f"`{m.get('source','?')}` — page {m.get('page','?')} (chunk `{m.get('chunk_id','?')}`)"
        for m in metas
    ]
    return docs, citations


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
    col_back, _ = st.columns([1, 11])
    with col_back:
        st.markdown('<div class="btn-back"></div>', unsafe_allow_html=True)
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
        files_to_index = [
            uf for uf in uploaded_files
            if not st.session_state.get(f"cqa_indexed_{uf.name}_{uf.size}")
        ]

        if files_to_index:
            with st.spinner(f"Indexing {len(files_to_index)} file(s)… this may take a moment."):
                total_added = 0
                for uf in files_to_index:
                    n = _build_index_from_bytes(uf.read(), uf.name, openai_client)
                    total_added += n
                    st.session_state[f"cqa_indexed_{uf.name}_{uf.size}"] = True

            if total_added > 0:
                st.success(f"Indexed {total_added} chunk(s) from {len(files_to_index)} file(s). Ready to chat!")
            else:
                st.info("No new chunks were added (files may already be indexed or contained no extractable text).")
        else:
            st.info(f"{len(uploaded_files)} file(s) already indexed this session.")

    # ── CHAT INTERFACE ────────────────────────────────────────────────────────
    doc_count = _store_count()
    if doc_count == 0:
        st.divider()
        st.warning(
            "**No documents indexed yet.**\n\n"
            "Upload PDFs or text files above to get started."
        )
        return

    st.divider()
    st.caption(f"Index ready — {doc_count} chunks from your uploaded documents.")

    st.info(
        "**Tip:** Ask specific questions about topics, concepts, or facts covered in your documents — "
        "e.g. *\"What is the definition of X?\"*, *\"Explain how Y works\"*, or *\"What does the document say about Z?\"* "
        "Vague questions like \"tell me about the file\" may not return useful answers."
    )

    if "qa_messages" not in st.session_state:
        st.session_state.qa_messages = []

    for msg in st.session_state.qa_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a specific question about your course materials…")
    if question:
        st.session_state.qa_messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents…"):
                docs, citations = _retrieve(question, openai_client)
                answer = _answer(question, docs, openai_client)

            response = answer
            if citations and NOT_FOUND_MSG not in answer:
                response += "\n\n**Sources:**\n" + "\n".join(f"- {c}" for c in citations)

            st.markdown(response)

        st.session_state.qa_messages.append({"role": "assistant", "content": response})
