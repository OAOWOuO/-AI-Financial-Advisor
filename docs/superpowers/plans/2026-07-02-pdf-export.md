# PDF Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "📄 Export PDF Report" download button that generates a 6-section professional financial analysis PDF from already-computed session data.

**Architecture:** New `sa_pdf.py` module with a single `generate_pdf(data, valuation) -> bytes` function using `fpdf2`. `stock_analyzer.py` calls it from a `st.download_button` in the left column. No new API calls — all data is already in the analysis result dict.

**Tech Stack:** `fpdf2` (pure-Python PDF generation), `streamlit` `st.download_button`, existing `sa_peers.build_peer_table`.

---

## File Map

| File | Change |
|------|--------|
| `streamlit_app/sa_pdf.py` | **New** — `generate_pdf(data, valuation) -> bytes`, all PDF sections |
| `streamlit_app/stock_analyzer.py` | Add `import datetime`, `from sa_pdf import generate_pdf`; add `st.download_button` in left column |
| `requirements.txt` | Add `fpdf2>=2.8.0` |
| `tests/test_pdf.py` | **New** — 5 unit tests (no network, no API) |

---

## Task 1: Create `sa_pdf.py` and `tests/test_pdf.py`

**Files:**
- Create: `streamlit_app/sa_pdf.py`
- Create: `tests/test_pdf.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pdf.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))

from sa_pdf import generate_pdf

_DUMMY_DATA = {
    "valid": True, "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology",
    "price": 200.0, "pe_ratio": 29.0, "forward_pe": 25.0, "eps": 6.43,
    "revenue_growth": 0.08, "profit_margin": 0.25, "market_cap": 3e12,
    "debt_to_equity": 1.5,
    "info": {"sharesOutstanding": 15_500_000_000, "ebitda": 130_000_000_000,
             "totalDebt": 110_000_000_000, "totalCash": 165_000_000_000},
    "_orchestrator_thesis": "Apple Inc. demonstrates strong financial health with robust cash generation.",
    "_valuation": {
        "intrinsic_value": 180.0,
        "ev_ebitda": 22.7,
        "summary": "Based on DCF analysis, AAPL appears modestly overvalued at current prices.",
    },
    "_peer_data": [],
    "_insider_signal_text": "Recent insider activity shows modest buying from C-suite executives.",
}

_DUMMY_VALUATION = {
    "pe_valuation": {"low": 150.0, "mid": 180.0, "high": 210.0},
    "forward_pe_valuation": {"low": 155.0, "mid": 185.0, "high": 215.0},
}


def test_generate_pdf_returns_bytes():
    result = generate_pdf(_DUMMY_DATA, _DUMMY_VALUATION)
    assert isinstance(result, bytes)
    assert len(result) > 1000


def test_generate_pdf_starts_with_pdf_header():
    result = generate_pdf(_DUMMY_DATA, _DUMMY_VALUATION)
    assert result[:4] == b"%PDF"


def test_generate_pdf_no_peer_data():
    data = dict(_DUMMY_DATA)
    data["_peer_data"] = []
    result = generate_pdf(data, _DUMMY_VALUATION)
    assert result[:4] == b"%PDF"


def test_generate_pdf_no_valuation_data():
    data = dict(_DUMMY_DATA)
    data["_valuation"] = {}
    result = generate_pdf(data, {})
    assert result[:4] == b"%PDF"


def test_generate_pdf_no_orchestrator_thesis():
    data = dict(_DUMMY_DATA)
    data.pop("_orchestrator_thesis", None)
    result = generate_pdf(data, _DUMMY_VALUATION)
    assert result[:4] == b"%PDF"
```

- [ ] **Step 2: Run to confirm ImportError**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/test_pdf.py::test_generate_pdf_returns_bytes -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'generate_pdf'`

- [ ] **Step 3: Create `streamlit_app/sa_pdf.py`**

```python
import datetime
from typing import Dict, Optional

from fpdf import FPDF


class _PDF(FPDF):
    def __init__(self, ticker: str, date_str: str):
        super().__init__(orientation="P", unit="mm", format="Letter")
        self._ticker = ticker
        self._date_str = date_str
        self.set_margins(left=18, top=18, right=18)
        self.set_auto_page_break(auto=True, margin=18)
        self.alias_nb_pages()

    def footer(self):
        self.set_y(-13)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(
            0, 5,
            f"AI Financial Advisor — {self._ticker} — {self._date_str}  |  Page {self.page_no()}/{{nb}}",
            align="C",
        )
        self.set_text_color(0, 0, 0)


def _fmt(val, fmt: str = "{:.1f}", suffix: str = "", prefix: str = "", scale: float = 1.0) -> str:
    if val is None:
        return "N/A"
    try:
        return f"{prefix}{fmt.format(float(val) * scale)}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def _market_cap_str(val) -> str:
    if not val:
        return "N/A"
    if val >= 1e12:
        return f"${val / 1e12:.2f}T"
    return f"${val / 1e9:.1f}B"


def _section_heading(pdf: FPDF, title: str) -> None:
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(26, 58, 92)
    pdf.cell(0, 6, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(26, 58, 92)
    pdf.set_line_width(0.5)
    pdf.line(pdf.get_x(), pdf.get_y(), pdf.get_x() + pdf.epw, pdf.get_y())
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)


def _two_col_table(pdf: FPDF, rows: list) -> None:
    label_w = 65
    val_w = pdf.epw - label_w
    for i, (label, value) in enumerate(rows):
        fill_color = (245, 247, 250) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*fill_color)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(label_w, 6, label, fill=True)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(val_w, 6, value, fill=True, new_x="LMARGIN", new_y="NEXT")


def generate_pdf(data: Dict, valuation: Dict) -> bytes:
    """Generate a PDF financial analysis report. Returns raw PDF bytes."""
    ticker = data.get("ticker", "N/A")
    name = data.get("name", ticker)
    date_str = datetime.date.today().isoformat()

    pdf = _PDF(ticker, date_str)
    pdf.add_page()

    # ── Section 1: Header ────────────────────────────────────────────────────
    half_w = pdf.epw / 2
    pdf.set_fill_color(26, 58, 92)
    pdf.set_text_color(255, 255, 255)

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(half_w, 10, ticker, fill=True)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(half_w, 10, "AI Financial Analysis Report", fill=True, align="R",
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(half_w, 7, name, fill=True)
    pdf.cell(half_w, 7, date_str, fill=True, align="R",
             new_x="LMARGIN", new_y="NEXT")

    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    price = _fmt(data.get("price"), "{:.2f}", prefix="$")
    mcap = _market_cap_str(data.get("market_cap"))
    sector = data.get("sector") or "N/A"
    pe = _fmt(data.get("pe_ratio"), "{:.1f}", suffix="x")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 5, f"Price: {price}   |   Market Cap: {mcap}   |   Sector: {sector}   |   P/E: {pe}",
             new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    # ── Section 2: AI Investment Thesis ──────────────────────────────────────
    _section_heading(pdf, "AI Investment Thesis")
    thesis = (data.get("_orchestrator_thesis") or "").strip() or "Analysis not available."
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, thesis)

    # ── Section 3: Fundamental Metrics ───────────────────────────────────────
    _section_heading(pdf, "Fundamental Metrics")
    _two_col_table(pdf, [
        ("Current Price",   _fmt(data.get("price"), "{:.2f}", prefix="$")),
        ("Market Cap",      _market_cap_str(data.get("market_cap"))),
        ("P/E (TTM)",       _fmt(data.get("pe_ratio"), "{:.1f}", suffix="x")),
        ("Forward P/E",     _fmt(data.get("forward_pe"), "{:.1f}", suffix="x")),
        ("EPS (TTM)",       _fmt(data.get("eps"), "{:.2f}", prefix="$")),
        ("Revenue Growth",  _fmt(data.get("revenue_growth"), "{:+.1f}", suffix="%", scale=100)),
        ("Profit Margin",   _fmt(data.get("profit_margin"), "{:.1f}", suffix="%", scale=100)),
        ("Debt / Equity",   _fmt(data.get("debt_to_equity"), "{:.2f}", suffix="x")),
    ])

    # ── Section 4: Valuation Summary ─────────────────────────────────────────
    _section_heading(pdf, "Valuation Summary")
    _val = data.get("_valuation") or {}
    intrinsic = _val.get("intrinsic_value")
    ev_ebitda_v = _val.get("ev_ebitda")
    narrative = (_val.get("summary") or "").strip()
    cur_price = data.get("price")

    if intrinsic and cur_price:
        margin = (intrinsic - cur_price) / intrinsic * 100
        margin_str = f"{margin:+.1f}% ({'Undervalued' if margin > 0 else 'Overvalued'})"
    else:
        margin_str = "N/A"

    if intrinsic or ev_ebitda_v:
        _two_col_table(pdf, [
            ("DCF Intrinsic Value", _fmt(intrinsic, "{:.2f}", prefix="$") + " / share"),
            ("Current Price",       _fmt(cur_price, "{:.2f}", prefix="$")),
            ("Margin of Safety",    margin_str),
            ("EV / EBITDA",         _fmt(ev_ebitda_v, "{:.1f}", suffix="x")),
        ])
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Valuation agent data unavailable.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    if narrative:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(0, 4, narrative)
        pdf.set_text_color(0, 0, 0)

    pe_val = (valuation or {}).get("pe_valuation")
    fpe_val = (valuation or {}).get("forward_pe_valuation")
    if pe_val or fpe_val:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "P/E-Based Price Targets:", new_x="LMARGIN", new_y="NEXT")
        target_rows = []
        if pe_val:
            target_rows.append((
                "P/E  Bear / Base / Bull",
                f"${pe_val['low']:.2f} / ${pe_val['mid']:.2f} / ${pe_val['high']:.2f}",
            ))
        if fpe_val:
            target_rows.append((
                "Fwd P/E  Bear / Base / Bull",
                f"${fpe_val['low']:.2f} / ${fpe_val['mid']:.2f} / ${fpe_val['high']:.2f}",
            ))
        _two_col_table(pdf, target_rows)
    else:
        pdf.ln(2)
        pdf.set_font("Helvetica", "I", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, "Insufficient data for P/E target calculation.", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

    # ── Section 5: Peer Comparison ────────────────────────────────────────────
    _section_heading(pdf, "Peer Comparison")
    peer_list = data.get("_peer_data") or []
    if not peer_list:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 5, "No peer data. Run analysis with an API key to populate peers.",
                 new_x="LMARGIN", new_y="NEXT")
    else:
        from sa_peers import build_peer_table
        peer_df = build_peer_table(ticker, data, peer_list)

        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 4, "Source: Yahoo Finance · GPT-4o", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)

        col_names = list(peer_df.columns)
        col_widths = [20, 20, 16, 19, 22, 20, 22, 18, 18]

        # Header row
        pdf.set_font("Helvetica", "B", 7)
        pdf.set_fill_color(26, 58, 92)
        pdf.set_text_color(255, 255, 255)
        for col, w in zip(col_names, col_widths):
            pdf.cell(w, 5, col, fill=True)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

        # Data rows
        for i, (_, row) in enumerate(peer_df.iterrows()):
            is_main = str(row["Ticker"]) == str(ticker)
            if is_main:
                pdf.set_fill_color(221, 238, 255)
            elif i % 2 == 0:
                pdf.set_fill_color(245, 247, 250)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.set_font("Helvetica", "B" if is_main else "", 7)
            for col, w in zip(col_names, col_widths):
                val = row[col]
                if val is None:
                    cell_text = "N/A"
                elif col == "Ticker":
                    cell_text = str(val)
                else:
                    try:
                        cell_text = f"{float(val):.1f}"
                    except (TypeError, ValueError):
                        cell_text = str(val)
                pdf.cell(w, 5, cell_text, fill=True)
            pdf.ln()

    # ── Section 6: Insider Activity ───────────────────────────────────────────
    _section_heading(pdf, "Insider Activity")
    insider_text = (data.get("_insider_signal_text") or "").strip() or "No insider data available."
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, insider_text)

    return bytes(pdf.output())
```

- [ ] **Step 4: Run all 5 tests**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/test_pdf.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Compile check**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m py_compile streamlit_app/sa_pdf.py && echo "Compile OK"
```

- [ ] **Step 6: Commit**

```bash
cd /Users/oaowouo/AI-Financial-Advisor && git add streamlit_app/sa_pdf.py tests/test_pdf.py && git commit -m "feat: add sa_pdf.py with generate_pdf and 5 unit tests"
```

---

## Task 2: Wire export button into `stock_analyzer.py` and update `requirements.txt`

**Files:**
- Modify: `streamlit_app/stock_analyzer.py` — 2 targeted edits
- Modify: `requirements.txt` — 1 line added

- [ ] **Step 1: Add `fpdf2` to `requirements.txt`**

Open `requirements.txt`. It currently ends with `python-dotenv>=1.0.0`. Add one line:

```
fpdf2>=2.8.0
```

Final `requirements.txt`:
```
streamlit>=1.45.0
pandas>=2.2.0
numpy>=2.1.0
altair>=5.5.0
yfinance>=1.1.0
openai>=2.24.0
pypdf>=6.7.0
python-dotenv>=1.0.0
fpdf2>=2.8.0
```

- [ ] **Step 2: Add imports to `stock_analyzer.py`**

Find the existing import block at the top of `stock_analyzer.py` (lines 1–7):

```python
import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, List, Tuple
```

Replace with:

```python
import os
import json
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, List, Tuple
from sa_pdf import generate_pdf
```

- [ ] **Step 3: Add the download button in the left column**

Find this exact text in `stock_analyzer.py` (around line 1886):

```python
        # ── AI CHAT FORM ─────────────────────────────────────────────────────────
        _chat_key = f"chat_history_{data['ticker']}" if has_data else "chat_history_default"
```

Insert the following block BEFORE it (preserve the 8-space indentation):

```python
        # ── PDF EXPORT ───────────────────────────────────────────────────────────
        if has_data and valuation is not None:
            _pdf_bytes = generate_pdf(data, valuation)
            st.download_button(
                label="📄 Export PDF Report",
                data=_pdf_bytes,
                file_name=f"{data['ticker']}_analysis_{datetime.date.today()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

        # ── AI CHAT FORM ─────────────────────────────────────────────────────────
        _chat_key = f"chat_history_{data['ticker']}" if has_data else "chat_history_default"
```

(This replaces the original `# ── AI CHAT FORM` line with the new block followed by the original line.)

- [ ] **Step 4: Compile check**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m py_compile streamlit_app/stock_analyzer.py && echo "Compile OK"
```

- [ ] **Step 5: Run full test suite**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/ -v 2>&1 | tail -10
```

Expected: 189 PASSED (184 existing + 5 pdf tests)

- [ ] **Step 6: Commit**

```bash
cd /Users/oaowouo/AI-Financial-Advisor && git add streamlit_app/stock_analyzer.py requirements.txt && git commit -m "feat: add PDF export button to stock analyzer left column"
```

---

## Task 3: Final CI check

- [ ] **Step 1: Compile all modified files**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m py_compile streamlit_app/sa_pdf.py streamlit_app/stock_analyzer.py && echo "All compile OK"
```

- [ ] **Step 2: Run full test suite**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/ -v 2>&1 | tail -10
```

Expected: 189 PASSED

- [ ] **Step 3: Push**

```bash
git push origin main
```

---

## Self-Review Checklist

| Spec requirement | Task |
|---|---|
| `generate_pdf(data, valuation) -> bytes` | Task 1 |
| Header: ticker, name, date, sector/price stats bar | Task 1 |
| AI Investment Thesis section from `_orchestrator_thesis` | Task 1 |
| Fundamental Metrics two-column table (8 rows) | Task 1 |
| Valuation: DCF intrinsic value, margin of safety, EV/EBITDA | Task 1 |
| Valuation: P/E Bear/Base/Bull from `valuation` dict | Task 1 |
| Valuation narrative from `_val["summary"]` | Task 1 |
| Peer comparison table (9 cols, main row highlighted) | Task 1 |
| Peer table: None → "N/A", floats to 1 decimal | Task 1 |
| Insider activity section from `_insider_signal_text` | Task 1 |
| Footer: ticker + date + page number on every page | Task 1 |
| Fallback text for every missing data field | Task 1 |
| `st.download_button` in left column, shown when `has_data` | Task 2 |
| Filename: `{ticker}_analysis_{date}.pdf` | Task 2 |
| `fpdf2>=2.8.0` added to `requirements.txt` | Task 2 |
| `import datetime` added to `stock_analyzer.py` | Task 2 |
| 5 unit tests covering all fallback paths | Task 1 |
