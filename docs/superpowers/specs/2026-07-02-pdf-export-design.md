# PDF Export — Design Spec
Date: 2026-07-02
Project: AI Financial Advisor — Stock Analyzer

---

## Overview

Add a "📄 Export PDF Report" download button to the Stock Analyzer. When clicked, it generates a structured financial analysis PDF from the data already in session state — no additional API calls. The PDF is returned as bytes via `st.download_button`, triggering a browser download without a page reload.

---

## 1. New File: `streamlit_app/sa_pdf.py`

Single public function:

```python
def generate_pdf(data: Dict, valuation: Dict) -> bytes
```

- `data`: the `final_data` dict from `run_multi_agent_research`
- `valuation`: the dict returned by `calculate_valuation(data, fund_analysis)` in `stock_analyzer.py`
- Returns: raw PDF bytes

### PDF Library

`fpdf2` (already installed). Portrait letter (215.9 × 279.4 mm), 18mm margins all sides. Font: Helvetica (built-in core font, no file needed). No images/charts — text and tables only.

### Page Sections (in order)

#### Section 1 — Header

Full-width dark-blue bar (`#1a3a5c`), white text:
- Left: ticker symbol (18pt bold) + company name (11pt)
- Right: "AI Financial Analysis Report" + date (10pt)

Below bar, one row of key stats (11pt gray):
`Price | Market Cap | Sector | P/E`

#### Section 2 — AI Investment Thesis

Heading "AI Investment Thesis" (12pt bold, dark blue underline).
Body: `data.get("_orchestrator_thesis", "")` wrapped to page width (10pt). If empty, print "Analysis not available."

#### Section 3 — Key Metrics Table

Heading "Fundamental Metrics" (12pt bold).
Two-column label/value table, alternating gray rows:

| Metric | Source |
|--------|--------|
| Current Price | `data.get("price")` → "$X.XX" |
| Market Cap | `data.get("market_cap")` → "$X.XB" |
| P/E (TTM) | `data.get("pe_ratio")` → "Xx" |
| Forward P/E | `data.get("forward_pe")` → "Xx" |
| EPS (TTM) | `data.get("eps")` → "$X.XX" |
| Revenue Growth | `data.get("revenue_growth")` → "+X.X%" |
| Profit Margin | `data.get("profit_margin")` → "X.X%" |
| Debt / Equity | `data.get("debt_to_equity")` → "X.Xx" |

"N/A" for any missing value.

#### Section 4 — Valuation Summary

Heading "Valuation Summary" (12pt bold).
Two sub-sections:

**A. DCF / Agent Valuation** (from `data.get("_valuation") or {}`):
- Intrinsic Value (DCF): `_val.get("intrinsic_value")` → "$X.XX / share"
- Current Price: `data.get("price")`
- Margin of Safety: `(intrinsic - price) / intrinsic * 100` → "+X.X% (Undervalued)" or "-X.X% (Overvalued)"
- EV/EBITDA: `_val.get("ev_ebitda")` → "Xx"
- Narrative: `_val.get("summary", "")` (wrapped text, 9pt, gray)

**B. P/E-Based Targets** (from `valuation` dict):
- If `valuation.get("pe_valuation")`: Bear/Base/Bull prices
- If `valuation.get("forward_pe_valuation")`: Bear/Base/Bull prices
- If neither: "Insufficient data for P/E target calculation."

#### Section 5 — Peer Comparison Table

Heading "Peer Comparison" (12pt bold).
Caption: "Source: Yahoo Finance · GPT-4o" (9pt gray).

If `data.get("_peer_data")`:
  Build DataFrame via `build_peer_table(data["ticker"], data, data["_peer_data"])`.
  Render as table — 9 columns, 8pt font, narrow columns (see widths below).
  Main ticker row: light blue background (`#ddeeff`). Others: alternating white / light gray.

Column widths (mm, total 175mm):
`Ticker=20, Market Cap=20, P/E=16, Fwd P/E=19, EV/EBITDA=22, Rev G%=20, Margin%=22, EPS=18, D/E=18`

Format None → "N/A". Floats to 1 decimal place.

If no peer data: "No peer data. Run analysis with an API key to populate peers."

#### Section 6 — Insider Activity

Heading "Insider Activity" (12pt bold).
Body: `data.get("_insider_signal_text", "")` or "No insider data available."

#### Footer

Every page: "AI Financial Advisor — {ticker} — {date}" centered, 8pt gray. Page number right-aligned.

---

## 2. `stock_analyzer.py` Changes

In the left column (`with col_left:`), after the insider weight slider block and before the AI chat form, add:

```python
if has_data and valuation is not None:
    _pdf_bytes = generate_pdf(data, valuation)
    st.download_button(
        label="📄 Export PDF Report",
        data=_pdf_bytes,
        file_name=f"{data['ticker']}_analysis_{datetime.date.today()}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
```

Import at top of file: `from sa_pdf import generate_pdf` and `import datetime` (if not already present).

---

## 3. `requirements.txt` Change

Add: `fpdf2>=2.8.0`

---

## 4. Tests: `tests/test_pdf.py`

5 unit tests — no network, no API key:

- `test_generate_pdf_returns_bytes` — output is `bytes`, non-empty
- `test_generate_pdf_starts_with_pdf_header` — output starts with `b"%PDF"`
- `test_generate_pdf_no_peer_data` — works when `_peer_data` is absent
- `test_generate_pdf_no_valuation_data` — works when `_valuation` is `{}`
- `test_generate_pdf_no_orchestrator_thesis` — works when `_orchestrator_thesis` is absent

---

## 5. Files Changed

| File | Change |
|------|--------|
| `streamlit_app/sa_pdf.py` | **New** — `generate_pdf(data, valuation) -> bytes` |
| `streamlit_app/stock_analyzer.py` | Add `generate_pdf` import; add `st.download_button` in left column |
| `requirements.txt` | Add `fpdf2>=2.8.0` |
| `tests/test_pdf.py` | **New** — 5 unit tests |

---

## 6. Fallback Table

| Situation | Behaviour |
|-----------|-----------|
| No `_orchestrator_thesis` | Print "Analysis not available." |
| No `_valuation` | Skip DCF section, print "Valuation agent data unavailable." |
| No `_peer_data` | Print "No peer data." |
| No `_insider_signal_text` | Print "No insider data available." |
| P/E valuation missing | Print "Insufficient data for P/E target calculation." |
| Any unexpected exception in `generate_pdf` | Propagate — `st.download_button` won't render, no crash |
