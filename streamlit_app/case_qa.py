"""
Case Q&A — RAG chat grounded in course / case materials.

Answers questions using ONLY documents in data/raw/, with citations.
Refuses to answer when the indexed documents don’t support the question.
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

INDEX_DIR = Path("index")
COLLECTION_NAME = "case_materials"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 5
# Cosine distance threshold: 0 = identical, 2 = opposite.
# Reject chunks whose distance exceeds this value.
DISTANCE_THRESHOLD = 1.0

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


# ────────────────────────  helpers  ────────────────────────

def _get_collection():
    """Load ChromaDB collection; return None if index not yet built."""
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(INDEX_DIR))
        return client.get_collection(COLLECTION_NAME)
    except Exception:
        return None


def _embed(text: str, openai_client) -> list[float]:
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def _retrieve(
    question: str, collection, openai_client
) -> tuple[list[str], list[str]]:
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
            Answers are grounded <em>only</em> in the documents you place in
            <code>data/raw/</code> — with file &amp; page citations.
            Unsupported questions are refused explicitly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Guard: index must exist
    index_ready = INDEX_DIR.exists() and any(INDEX_DIR.iterdir())
    if not index_ready:
        st.warning(
            "**No index found.**\n\n"
            "Drop PDFs or `.md` files into `data/raw/`, then run:\n\n"
            "```bash\npython scripts/build_index.py\n```"
        )
        return

    collection = _get_collection()
    if collection is None:
        st.error(
            "Failed to load the index. "
            "Try rebuilding it with `python scripts/build_index.py`."
        )
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("**OPENAI_API_KEY** not set. Add it to your `.env` file.")
        return

    import openai
    openai_client = openai.OpenAI(api_key=api_key)

    st.caption(f"Index ready — {collection.count()} chunks from your documents.")
    st.divider()

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
