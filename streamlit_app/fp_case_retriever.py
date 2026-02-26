"""
fp_case_retriever.py — Built-in case library retriever.

Loads pre-built MD narrative + structured JSON summaries from data/cases/.
Embeds them once per Streamlit session, then returns semantically similar cases
for any given ClientProfile + issues list.

Architecture:
  - Structured summaries (JSON) → fast rule-based pre-filter
  - Raw narrative (MD) → embedding similarity (hybrid)
  - Deterministic "why matched" reasons (no extra LLM call)
"""
from __future__ import annotations
import json, os, logging
from typing import List, Dict, Any, Optional

import streamlit as st
import numpy as np

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────

_BASE       = os.path.join(os.path.dirname(__file__), "data", "cases")
RAW_DIR     = os.path.join(_BASE, "raw")
RAW_EXT_DIR = os.path.join(_BASE, "raw_external")   # gitignored PDFs
STRUCT_DIR  = os.path.join(_BASE, "structured")
INDEX_PATH  = os.path.join(_BASE, "index", "case_index.json")
EMBED_MODEL = "text-embedding-3-small"

_CAT_TO_TOPIC = {
    "EMERGENCY_FUND": "cash_flow",
    "DEBT":           "debt",
    "INSURANCE":      "insurance",
    "RETIREMENT":     "retirement",
    "TAX":            "tax",
    "ESTATE":         "estate",
    "INVESTMENT":     "investing",
    "GOALS":          "goals",
    "CASH_FLOW":      "cash_flow",
    "NET_WORTH":      "investing",
}


# ── Session-level store ────────────────────────────────────────────────────────

def _get_store() -> Optional[Dict]:
    return st.session_state.get("fp_case_store")

def _set_store(store: Dict) -> None:
    st.session_state["fp_case_store"] = store

def is_indexed() -> bool:
    store = _get_store()
    return store is not None and store.get("has_embeddings", False)

def clear_case_store() -> None:
    if "fp_case_store" in st.session_state:
        del st.session_state["fp_case_store"]


# ── File loaders ──────────────────────────────────────────────────────────────

def load_case_index() -> List[Dict]:
    if not os.path.exists(INDEX_PATH):
        return []
    with open(INDEX_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_structured_summary(case_id: str) -> Dict:
    path = os.path.join(STRUCT_DIR, f"{case_id}_summary.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_raw_case(case_id: str) -> str:
    """Return raw narrative text.  MD file for internal cases; raw_text_excerpt from JSON for external."""
    path = os.path.join(RAW_DIR, f"{case_id}.md")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    # Fallback: use the raw_text_excerpt stored in the structured summary (external PDF cases)
    structured = load_structured_summary(case_id)
    return structured.get("raw_text_excerpt", "")

def is_external(case_id: str) -> bool:
    """True if the case comes from an external PDF source (not a built-in MD narrative)."""
    return not os.path.exists(os.path.join(RAW_DIR, f"{case_id}.md"))

def case_count() -> int:
    return len(load_case_index())

def case_count_by_source() -> Dict[str, int]:
    """Return {source_type: count} for Settings display."""
    counts: Dict[str, int] = {}
    for entry in load_case_index():
        st_ = entry.get("source_type", "internal")
        counts[st_] = counts.get(st_, 0) + 1
    return counts


# ── Embedding builder ─────────────────────────────────────────────────────────

def build_case_embeddings(openai_client) -> Dict:
    """
    Load all cases, embed them, and cache in session state.
    Returns the store dict. Idempotent — only embeds once per session.
    """
    existing = _get_store()
    if existing is not None:
        return existing

    index = load_case_index()
    if not index:
        store = {"cases": [], "has_embeddings": False,
                 "error": "Case index not found at data/cases/index/case_index.json"}
        _set_store(store)
        logger.warning("Case index missing: %s", INDEX_PATH)
        return store

    cases = []
    for meta in index:
        case_id    = meta["case_id"]
        raw        = load_raw_case(case_id)
        structured = load_structured_summary(case_id)
        embed_text = _build_embed_text(meta, structured, raw)
        cases.append({
            "case_id":    case_id,
            "metadata":   meta,
            "structured": structured,
            "raw":        raw,
            "embed_text": embed_text,
            "embedding":  None,
        })

    logger.info("Embedding %d cases from built-in library…", len(cases))

    texts = [c["embed_text"] for c in cases]
    try:
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
        embeddings = [r.embedding for r in resp.data]
        for i, case in enumerate(cases):
            case["embedding"] = embeddings[i]
        store = {"cases": cases, "has_embeddings": True,
                 "count": len(cases), "error": None}
        logger.info("Case library indexed: %d cases, %d embeddings", len(cases), len(embeddings))
    except Exception as e:
        store = {"cases": cases, "has_embeddings": False,
                 "count": len(cases), "error": str(e)}
        logger.error("Case embedding failed: %s", e)

    _set_store(store)
    return store


def _build_embed_text(meta: Dict, structured: Dict, raw: str) -> str:
    parts = [
        f"Case: {meta.get('title', '')}",
        f"Household type: {meta.get('household_type', '')}",
        f"Life stage: {meta.get('life_stage', '')}",
        f"Topics: {', '.join(meta.get('major_topics', []))}",
        f"Key issues: {'; '.join(meta.get('key_issues', []))}",
    ]
    if structured:
        issues = structured.get("planning_issues", [])
        # Support both internal (candidate_recommendations) and external (key_recommendations) formats
        recs = structured.get("candidate_recommendations") or structured.get("key_recommendations", [])
        if issues:
            issue_texts = [i if isinstance(i, str) else i.get("description", "") for i in issues[:6]]
            parts.append("Planning issues: " + "; ".join(issue_texts))
        if recs:
            parts.append("Recommendations: " + "; ".join(recs[:5]))
    # First 600 chars of raw narrative / text excerpt for semantic richness
    if raw:
        parts.append(raw[:600])
    return "\n".join(parts)


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_similar_cases(
    profile,           # fp_schemas.ClientProfile
    issues: List[Any], # List[PlanningIssue]
    openai_client,
    top_k: int = 3,
    mode: str = "hybrid",  # "structured" | "hybrid"
) -> List[Dict]:
    """
    Return top_k cases most similar to this client's profile + issues.
    Each result: {case_id, metadata, structured, raw_excerpt, score, reasons, score_pct}
    Falls back to rule-based matching if embeddings unavailable.
    """
    store = build_case_embeddings(openai_client)
    cases = store.get("cases", [])

    if not cases:
        return []

    if not store.get("has_embeddings") or mode == "structured":
        return _rule_based_match(profile, issues, cases, top_k)

    # Embed query
    query = _build_query(profile, issues)
    try:
        resp = openai_client.embeddings.create(model=EMBED_MODEL, input=query)
        q = np.array(resp.data[0].embedding, dtype="float32")
        q /= np.linalg.norm(q) + 1e-10
    except Exception as e:
        logger.error("Query embedding failed: %s", e)
        return _rule_based_match(profile, issues, cases, top_k)

    scored = []
    for case in cases:
        if case.get("embedding") is None:
            continue
        e = np.array(case["embedding"], dtype="float32")
        e /= np.linalg.norm(e) + 1e-10
        sim = float(q @ e)
        reasons = _explain_match(profile, issues, case)
        scored.append(_make_result(case, round(sim, 4), reasons))

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _make_result(case: Dict, score: float, reasons: List[str]) -> Dict:
    """Build a standardised result dict with source/citation fields."""
    meta = case["metadata"]
    structured = case["structured"]
    source_type = meta.get("source_type") or structured.get("source_type") or "internal"
    source_file = meta.get("source_file") or structured.get("source_file") or ""
    source_pages = meta.get("source_pages") or structured.get("source_pages") or []

    # Build human-readable citation string
    if source_file and source_pages:
        pg_str = f"pp. {source_pages[0]}–{source_pages[-1]}" if len(source_pages) > 1 else f"p. {source_pages[0]}"
        citation = f"{source_file}, {pg_str}"
    elif source_file:
        citation = source_file
    else:
        citation = "Built-in case library"

    return {
        "case_id":     case["case_id"],
        "metadata":    meta,
        "structured":  structured,
        "raw_excerpt": (case.get("raw") or "")[:800],
        "score":       round(score, 4),
        "score_pct":   round(score * 100, 1),
        "reasons":     reasons,
        "source_type": source_type,
        "source_file": source_file,
        "source_pages": source_pages,
        "citation":    citation,
    }


def _build_query(profile, issues) -> str:
    issue_str = "; ".join(iss.title for iss in issues[:6]) if issues else "general planning"
    return (
        f"Financial planning case: {profile.marital_status} individual, age {profile.age}, "
        f"{profile.dependents} dependents, income ${profile.total_annual_income():,.0f}/yr. "
        f"Planning issues: {issue_str}. "
        f"Risk tolerance: {profile.risk_tolerance}. "
        f"Net worth: ${profile.net_worth():,.0f}."
    )


def _explain_match(profile, issues, case: Dict) -> List[str]:
    """Deterministic similarity reasons — no LLM needed."""
    reasons = []
    meta = case["metadata"]

    # Age proximity
    age_range = meta.get("age_range", [])
    if len(age_range) == 2 and age_range[0] <= profile.age <= age_range[1]:
        reasons.append(f"Age match ({age_range[0]}–{age_range[1]} yrs)")

    # Household type
    if meta.get("household_type", "").lower() in profile.marital_status.lower():
        reasons.append(f"Same household type ({meta['household_type']})")

    # Topic overlap with issue categories
    case_topics = set(meta.get("major_topics", []))
    matched_topics = []
    for iss in (issues or []):
        topic = _CAT_TO_TOPIC.get(str(iss.category), "")
        if topic and topic in case_topics and topic not in matched_topics:
            matched_topics.append(topic)
    if matched_topics:
        reasons.append(f"Shared topics: {', '.join(matched_topics[:3])}")

    # Income range
    income_range = meta.get("income_range", [])
    income = profile.total_annual_income()
    if len(income_range) == 2 and income_range[0] <= income <= income_range[1]:
        reasons.append(f"Similar income bracket")

    # Life stage keyword matching
    situation = profile.situation_summary.lower() if profile.situation_summary else ""
    stage = meta.get("life_stage", "")
    stage_keywords = {
        "early_career":    ["student loan", "renting", "entry level", "early career", "recent grad"],
        "mid_career":      ["mid career", "growing family", "mortgage", "business owner"],
        "pre_retirement":  ["retirement", "10 years", "catch up", "pre-retirement"],
        "retirement":      ["retired", "rmd", "social security", "decumulation"],
        "family_formation":["kids", "children", "college", "529", "daycare"],
    }
    for kw in stage_keywords.get(stage, []):
        if kw in situation:
            reasons.append(f"Life stage keyword: '{kw}'")
            break

    if not reasons:
        reasons.append("Similar overall financial profile")

    return reasons[:4]


def _rule_based_match(profile, issues, cases, top_k) -> List[Dict]:
    """Fallback similarity when embeddings are unavailable."""
    issue_cats = {str(iss.category) for iss in (issues or [])}
    scored = []
    for case in cases:
        meta = case["metadata"]
        score = 0.0

        # Household type (0–0.3)
        if meta.get("household_type", "").lower() in profile.marital_status.lower():
            score += 0.30

        # Topic overlap (up to 0.40)
        case_topics = set(meta.get("major_topics", []))
        for cat in issue_cats:
            if _CAT_TO_TOPIC.get(cat, "") in case_topics:
                score += 0.10

        # Age range (0–0.20)
        ar = meta.get("age_range", [0, 100])
        if isinstance(ar, list) and len(ar) == 2 and ar[0] <= profile.age <= ar[1]:
            score += 0.20

        # Income range (0–0.10)
        ir = meta.get("income_range", [0, 1e9])
        if isinstance(ir, list) and len(ir) == 2 and ir[0] <= profile.total_annual_income() <= ir[1]:
            score += 0.10

        reasons = _explain_match(profile, issues, case)
        scored.append(_make_result(case, round(min(score, 1.0), 4), reasons))

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
