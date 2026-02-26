#!/usr/bin/env python3
"""
ingest_fpa_cases.py
-------------------
For each FPA FPC PDF in raw_external/:
  1. Extract text per page using pypdf
  2. Call OpenAI GPT-4o-mini to generate a structured summary
  3. Write to data/cases/structured/fpa_fpc_YYYY_summary.json
  4. Upsert entry in data/cases/index/case_index.json

The structured JSON files ARE committed (they contain metadata + text excerpt).
The raw PDFs remain gitignored (copyright material).

Usage:
    # From repo root — requires OPENAI_API_KEY in .env or env
    python scripts/ingest_fpa_cases.py
    python scripts/ingest_fpa_cases.py --year 2024        # single year
    python scripts/ingest_fpa_cases.py --force            # re-ingest all
"""

from __future__ import annotations
import argparse, json, os, pathlib, re, sys, textwrap
from typing import Optional

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO = pathlib.Path(__file__).parent.parent
STREAMLIT = REPO / "streamlit_app"
RAW_EXT   = STREAMLIT / "data" / "cases" / "raw_external"
STRUCT_DIR = STREAMLIT / "data" / "cases" / "structured"
INDEX_PATH = STREAMLIT / "data" / "cases" / "index" / "case_index.json"

# ── PDF catalogue (must match download_fpa_cases.py) ─────────────────────────
PDF_CATALOGUE = [
    ("FPA_FPC_2025.pdf", 2025, "https://www.financialplanningassociation.org/sites/default/files/2025-02/FPC25%20Case%20Study.pdf"),
    ("FPA_FPC_2024.pdf", 2024, "https://www.financialplanningassociation.org/sites/default/files/2025-02/FPC24%20Case%20Study%20Final%20Copy.pdf"),
    ("FPA_FPC_2023.pdf", 2023, "https://www.financialplanningassociation.org/sites/default/files/2023-02/FPC23%20Case%20Study%20-%20The%20Rogers%20Retire_FINAL.pdf"),
    ("FPA_FPC_2022.pdf", 2022, "https://www.financialplanningassociation.org/sites/default/files/2022-02/2022%20FPC%20Case%20Study%20Final.pdf"),
    ("FPA_FPC_2021.pdf", 2021, "https://www.financialplanningassociation.org/sites/default/files/2021-02/FPC21%20Case%20Study%20Final.pdf"),
    ("FPA_FPC_2020.pdf", 2020, "https://www.financialplanningassociation.org/sites/default/files/2020-05/FPC20%20Case%20Study.pdf"),
    ("FPA_FPC_2019.pdf", 2019, "https://www.financialplanningassociation.org/sites/default/files/2021-07/FPC%202019%20Case%20Study.pdf"),
    ("FPA_FPC_2018.pdf", 2018, "https://www.financialplanningassociation.org/sites/default/files/2021-07/FPC%202018%20Case%20Study.pdf"),
    ("FPA_FPC_2016.pdf", 2016, "https://www.financialplanningassociation.org/sites/default/files/2021-07/FPC%202016%20Case%20Study.pdf"),
    ("FPA_FPC_2015.pdf", 2015, "https://www.financialplanningassociation.org/sites/default/files/2021-07/FPC15%20Case%20Study.pdf"),
    ("FPA_FPC_2014.pdf", 2014, "https://www.financialplanningassociation.org/sites/default/files/2021-07/FPC%202014%20Case%20Study.pdf"),
]


# ── OpenAI client ─────────────────────────────────────────────────────────────
def _get_openai():
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or export it.")
    from openai import OpenAI
    return OpenAI(api_key=api_key)


# ── PDF extraction ────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_path: pathlib.Path) -> dict[int, str]:
    """Return {page_number: text} (1-indexed). Uses pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf")
    reader = PdfReader(str(pdf_path))
    pages = {}
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages[i] = text.strip()
    return pages


def build_full_text(pages: dict[int, str], max_chars: int = 12000) -> tuple[str, list[int]]:
    """Concatenate page text up to max_chars; return (text, pages_used)."""
    parts = []
    used = []
    total = 0
    for pg_num in sorted(pages.keys()):
        t = pages[pg_num]
        if not t:
            continue
        parts.append(f"[Page {pg_num}]\n{t}")
        used.append(pg_num)
        total += len(t)
        if total >= max_chars:
            break
    return "\n\n".join(parts), used


# ── LLM extraction ────────────────────────────────────────────────────────────
_EXTRACT_SYSTEM = """\
You are a CFP-certified financial planner summarising a real case study for a retrieval database.
Extract information ONLY from the provided text. Do not invent data.
Output strict JSON following the schema below exactly.
If a field cannot be determined from the text, use null (not a string).

Schema:
{
  "title": "<short descriptive title for the case, e.g. 'Rogers Family: Pre-Retirement and Estate Planning'>",
  "client_names": "<names of the main clients, e.g. 'Robert and Carol Rogers'>",
  "age_min": <integer or null>,
  "age_max": <integer or null>,
  "household_type": "<one of: single | married | married_with_children | single_parent | other>",
  "life_stage": "<one of: early_career | mid_career | pre_retirement | retirement | family_formation>",
  "income_min": <annual gross income lower bound in USD, integer or null>,
  "income_max": <annual gross income upper bound in USD, integer or null>,
  "major_topics": ["<topic1>", ...],
  "key_issues": ["<issue1>", ...],
  "planning_issues": ["<issue description 1>", ...],
  "key_recommendations": ["<recommendation 1>", ...],
  "outcome_summary": "<1-2 sentence outcome or null>",
  "key_lesson": "<1-2 sentence lesson for advisors or null>",
  "raw_text_excerpt": "<first 2500 characters of relevant case narrative text>"
}

For major_topics use only these values (can select multiple):
  debt, emergency_fund, insurance, retirement, tax, estate, investing, goals, cash_flow

For planning_issues: list 3-7 specific financial issues identified in the case with brief descriptions.
For key_recommendations: list 4-8 specific actionable recommendations.
"""


def extract_structured(client, full_text: str, year: int, filename: str) -> dict:
    user_msg = f"FPA FPC {year} Case Study ({filename}):\n\n{full_text[:12000]}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
        max_tokens=2000,
    )
    return json.loads(response.choices[0].message.content)


# ── Summary builder ───────────────────────────────────────────────────────────
def build_summary_json(
    filename: str,
    year: int,
    url: str,
    extracted: dict,
    pages_used: list[int],
) -> dict:
    case_id = f"fpa_fpc_{year}"
    title   = extracted.get("title") or f"FPA Financial Planning Competition {year}"
    age_min = extracted.get("age_min")
    age_max = extracted.get("age_max")
    inc_min = extracted.get("income_min")
    inc_max = extracted.get("income_max")

    return {
        "case_id":      case_id,
        "title":        title,
        "year":         year,
        "source_type":  "external_fpa_pdf",
        "source":       "FPA Financial Planning Competition",
        "source_file":  filename,
        "source_url":   url,
        "source_pages": pages_used[:10],
        "client_profile": {
            "name":         extracted.get("client_names", ""),
            "household_type": extracted.get("household_type", ""),
            "life_stage":   extracted.get("life_stage", ""),
        },
        "financial_snapshot": {
            "gross_income":     inc_min,
            "income_range_max": inc_max,
        },
        "age_range":    [age_min, age_max] if (age_min and age_max) else [],
        "income_range": [inc_min, inc_max] if (inc_min and inc_max) else [],
        "household_type": extracted.get("household_type", ""),
        "life_stage":   extracted.get("life_stage", ""),
        "major_topics": extracted.get("major_topics", []),
        "key_issues":   extracted.get("key_issues", []),
        "planning_issues":      extracted.get("planning_issues", []),
        "key_recommendations":  extracted.get("key_recommendations", []),
        "candidate_recommendations": extracted.get("key_recommendations", []),
        "outcome_summary": extracted.get("outcome_summary", ""),
        "key_lesson":      extracted.get("key_lesson", ""),
        "raw_text_excerpt": (extracted.get("raw_text_excerpt") or "")[:3000],
    }


# ── Index updater ─────────────────────────────────────────────────────────────
def upsert_index(summary: dict) -> None:
    if not INDEX_PATH.exists():
        entries = []
    else:
        with open(INDEX_PATH, encoding="utf-8") as f:
            entries = json.load(f)

    cid = summary["case_id"]
    entry = {
        "case_id":       cid,
        "title":         summary["title"],
        "year":          summary["year"],
        "source":        summary["source"],
        "source_type":   summary["source_type"],
        "source_file":   summary["source_file"],
        "source_url":    summary.get("source_url", ""),
        "source_pages":  summary.get("source_pages", []),
        "household_type": summary.get("household_type", ""),
        "life_stage":    summary.get("life_stage", ""),
        "age_range":     summary.get("age_range", []),
        "income_range":  summary.get("income_range", []),
        "major_topics":  summary.get("major_topics", []),
        "key_issues":    summary.get("key_issues", []),
        "outcome":       (summary.get("outcome_summary") or "")[:200],
    }

    # Replace existing or append
    updated = [e for e in entries if e.get("case_id") != cid]
    updated.append(entry)

    # Sort: internal cases first (case_XX), then FPA by year descending
    def sort_key(e):
        cid_ = e.get("case_id", "")
        if cid_.startswith("case_"):
            return (0, cid_)
        yr = e.get("year", 0)
        return (1, str(9999 - yr))

    updated.sort(key=sort_key)
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def ingest_one(client, filename: str, year: int, url: str, force: bool) -> bool:
    pdf_path = RAW_EXT / filename
    if not pdf_path.exists():
        print(f"  ✗ PDF not found: {filename} — run download_fpa_cases.py first")
        return False

    out_path = STRUCT_DIR / f"fpa_fpc_{year}_summary.json"
    if out_path.exists() and not force:
        print(f"  ✓ Already ingested: fpa_fpc_{year}_summary.json (use --force to re-ingest)")
        return True

    print(f"  → Ingesting FPC {year}: {filename} …")

    # Extract text
    pages = extract_pdf_text(pdf_path)
    full_text, pages_used = build_full_text(pages)
    print(f"     Extracted {len(pages)} pages, {len(full_text):,} chars; using pages {pages_used[:5]}…")

    # LLM extraction
    extracted = extract_structured(client, full_text, year, filename)
    title = extracted.get("title", f"FPA FPC {year}")
    print(f"     Title: {title}")
    print(f"     Topics: {extracted.get('major_topics', [])}")

    # Build summary
    summary = build_summary_json(filename, year, url, extracted, pages_used)

    # Write JSON
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"     Saved: {out_path.name}")

    # Update index
    upsert_index(summary)
    print(f"     Index updated.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Ingest FPA FPC case study PDFs into the case library")
    parser.add_argument("--year", type=int, help="Only ingest a specific year")
    parser.add_argument("--force", action="store_true", help="Re-ingest even if summary already exists")
    args = parser.parse_args()

    client = _get_openai()

    catalogue = PDF_CATALOGUE
    if args.year:
        catalogue = [(f, y, u) for f, y, u in catalogue if y == args.year]
        if not catalogue:
            print(f"Year {args.year} not found in catalogue.")
            sys.exit(1)

    print(f"\nIngesting {len(catalogue)} FPA FPC case studies…\n")
    ok = fail = skip = 0
    for filename, year, url in catalogue:
        result = ingest_one(client, filename, year, url, args.force)
        if result:
            ok += 1
        else:
            fail += 1

    print(f"\n{'─'*50}")
    print(f"Ingestion complete: {ok} succeeded, {fail} failed/skipped")
    print(f"Structured summaries → {STRUCT_DIR}")
    print(f"Case index updated  → {INDEX_PATH}")


if __name__ == "__main__":
    main()
