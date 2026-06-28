# Insider Trading (Form 4) Feature — Design Spec
Date: 2026-06-28
Project: AI Financial Advisor — Stock Analyzer

---

## Overview

Add SEC EDGAR Form 4 insider trading analysis to the Stock Analyzer. Insider open-market purchases by executives are one of the strongest documented alpha signals in academic literature. The feature adds a dedicated **👥 Insider** tab, contributes a new sub-score to the fundamental analysis, and exposes a user-adjustable weight slider in the sidebar.

---

## 1. Data Layer (`sa_research_agent.py`)

### `fetch_insider_trades(cik: str, months: int) -> Dict`

Fetches Form 4 filings from SEC EDGAR for the given company and time window.

**Source:** `https://data.sec.gov/submissions/CIK{cik}.json` → filter `form == "4"` within `months` → fetch each primary XML document.

**Filtering logic:**
- Keep only transaction codes `P` (open-market purchase) and `S` (open-market sale)
- Exclude `M` (option exercise), `F` (tax withholding), `A` (award) — these are not discretionary signals
- Max 15 Form 4 filings per call; 0.1s delay between requests (SEC rate limit compliance)

**Return schema:**
```python
{
    "available": bool,
    "error": str | None,         # set if EDGAR unavailable
    "months": int,               # time window used
    "trades": [
        {
            "date": "YYYY-MM-DD",
            "insider": str,      # name, title-cased
            "title": str,        # e.g. "CEO", "Director"
            "type": "BUY"|"SELL",
            "shares": int,
            "price": float,
            "value": float,      # shares * price
        }
    ],
    "summary": {
        "num_buys": int,
        "num_sells": int,
        "total_buy_value": float,
        "total_sell_value": float,
        "net_buy_value": float,
        "unique_buyers": int,
        "signal": "STRONG BUY SIGNAL"|"BUY SIGNAL"|"MIXED / NEUTRAL"|"SELL SIGNAL",
        "no_activity": bool,
    }
}
```

**Signal classification (deterministic):**
| Condition | Signal |
|-----------|--------|
| unique_buyers >= 3 AND net_buy_value > 0 | STRONG BUY SIGNAL |
| num_buys >= 1 AND net_buy_value > 0 | BUY SIGNAL |
| num_sells >= 3 AND net_buy_value < 0 | SELL SIGNAL |
| otherwise | MIXED / NEUTRAL |

### `analyze_insider_signal(trades_data: Dict, company_name: str, api_key: str) -> str`

Produces a 2–3 sentence natural-language interpretation.

- **With API key:** GPT-4o-mini prompt emphasises cluster buying, role of insiders (C-suite > director), magnitude vs typical compensation, and timing relative to earnings
- **Without API key:** deterministic template based on summary fields (e.g. "3 executives purchased $2.1M in shares over the past 6 months — historically a strong bullish signal.")
- Returns plain text string; never raises, falls back to deterministic on any exception

### Pipeline integration

Both pipeline mode (no LLM) and agent mode add insider trades as a mandatory final step:
- Step 3 in `_run_pipeline`
- Called after the agent loop in `run_research_agent`

Results stored in merged data dict:
- `data["_insider_trades"]` — raw result from `fetch_insider_trades`
- `data["_insider_signal_text"]` — string from `analyze_insider_signal`

Time window used for pipeline: **6 months** (default). User can change via UI radio button which re-fetches and re-scores.

---

## 2. Scoring Layer (`stock_analyzer.py`)

### `analyze_fundamentals(data, insider_weight_pct=15)`

New parameter `insider_weight_pct` (0–25, default 15) passed in from the sidebar slider.

**Insider sub-score (max points = `insider_weight_pct / 100 * 100`):**

| Condition | Raw score (out of 20) |
|-----------|-----------------------|
| unique_buyers >= 3 AND net_buy_value > $1M | 20 |
| num_buys >= 1 AND net_buy_value > 0 | 12 |
| No activity | 6 (neutral, no penalty for non-trading) |
| net_sell_value slightly > net_buy_value | 2 |
| num_sells >= 3 AND net_sell_value > $2M | −5 |

Raw score is then scaled: `insider_score = raw * (insider_weight_pct / 20)`

**Weight rebalancing:** The other four sub-categories (valuation, profitability, growth, health) are scaled by `(100 - insider_weight_pct) / 100` so the total always sums to 100 points.

Example at 15%:
| Sub-category | Old max | New max |
|---|---|---|
| Valuation | 30 | 25.5 |
| Profitability | 25 | 21.25 |
| Growth | 25 | 21.25 |
| Health | 20 | 17.0 |
| Insider | 0 | 15.0 |
| **Total** | **100** | **100** |

`generate_recommendation` is **not modified**. The change flows through naturally via the updated `fund_analysis` dict.

---

## 3. UI Layer (`stock_analyzer.py`)

### Sidebar slider

Appears after a successful analysis (inside `if has_data:`):
```
⚙️ Insider Signal Weight
[slider: 0–25%, default 15%, step 5%]
```
Stored in `st.session_state["insider_weight"]`. Changing it triggers `st.rerun()` which re-calls `analyze_fundamentals` with the new weight. No data re-fetch.

### New tab: 👥 Insider

Tab order becomes:
```
🏢 Profile | 📊 Technical | 📋 Fundamental | 👥 Insider | 🎯 Conclusion | 🔍 Research Log
```

**Tab content:**

1. **Time window selector** — radio buttons: `3M | 6M | 12M` (default 6M). Changing triggers re-fetch + re-score with spinner.

2. **Three metric cards:**
   - Net Insider Buying ($ value)
   - Unique Buyers (count)
   - Insider Score (e.g. "16 / 20")

3. **Trade table** — columns: Date, Insider, Title, Type (BUY/SELL color-coded), Shares, Value. Sorted newest first. Max 20 rows.

4. **AI Interpretation** — collapsible `st.expander("💬 AI Analysis")`, shows `_insider_signal_text`. Falls back to deterministic summary if no API key.

5. **Source caption** — "Source: SEC EDGAR Form 4 · {months}-month window · Open-market transactions only"

### Conclusion tab addition

Below the BUY/HOLD/SELL rating card, add one line:
```
Insider Signal: BULLISH  (Weight: 15%)
```
Color-coded: green for BUY/STRONG BUY, yellow for NEUTRAL, red for SELL.

### Fundamental tab

Remove the insider section (it now lives in its own tab). The Insider sub-score still appears in the score breakdown table row as "Insider Signal".

---

## 4. Error Handling

| Failure | Behaviour |
|---------|-----------|
| EDGAR returns non-200 | `available: False`; Insider tab shows "SEC EDGAR unavailable for this ticker" info box; insider weight auto-set to 0 and hidden from sidebar |
| Non-US ticker (no CIK) | Same as above |
| XML parse error on one filing | Skip that filing, continue with rest |
| LLM call fails | Fall back to deterministic summary silently |
| No Form 4 filings in window | Shows "No insider trading activity in the past {months} months." |

---

## 5. Files Changed

| File | Change |
|------|--------|
| `streamlit_app/sa_research_agent.py` | Add `fetch_insider_trades`, `analyze_insider_signal`, `_deterministic_insider_summary`; update `_run_pipeline`, `run_research_agent`, `_merge_data` |
| `streamlit_app/stock_analyzer.py` | Add `insider_weight_pct` param to `analyze_fundamentals`; add sidebar slider; add Insider tab; update Conclusion tab; update tab list from 5 to 6 |

No new dependencies required (uses `requests`, `xml.etree.ElementTree` — both stdlib/already present).
