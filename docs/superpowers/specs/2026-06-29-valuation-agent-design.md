# Valuation Agent — Design Spec
Date: 2026-06-29
Project: AI Financial Advisor — Stock Analyzer

---

## Overview

Add a fourth specialized sub-agent (`run_valuation_agent`) to the multi-agent architecture. It runs after the three parallel sub-agents complete, using the EDGAR data already fetched by `FundamentalAgent`. It computes three valuations (DCF, P/E comparison, EV/EBITDA) and returns a per-share intrinsic value estimate. A new 7th tab ("Valuation") displays the full breakdown. The Orchestrator receives a fourth report and incorporates valuation context into its thesis.

---

## 1. Execution Flow Change

```
Step 0:   yfinance base fetch
Steps 1-3: ThreadPoolExecutor — Fundamental + Catalyst + Macro (parallel, timeout=45s)
Step 3.5: run_valuation_agent — serial, uses Fundamental's edgar_data
Step 4:   run_orchestrator — receives 4 reports (fundamental, catalyst, macro, valuation)
Step 5:   fetch_insider_trades
→ _merge_data → return
```

`run_valuation_agent` is NOT added to `ThreadPoolExecutor` because it depends on `fundamental_report["edgar_data"]`. It runs immediately after the parallel block closes.

---

## 2. New Function: `run_valuation_agent`

### Signature

```python
def run_valuation_agent(
    ticker: str,
    yf_data: Dict,
    edgar_data: Dict,
    api_key: str,
) -> Dict
```

### Model
GPT-4o-mini

### Tools
- `get_yfinance_data` — to fetch any missing yfinance fields
- `get_sec_filing` — to verify FCF from EDGAR if `edgar_data` is empty
- Max 2 tool calls total

### System prompt focus
Compute three valuation metrics and give a 2-3 sentence conclusion on whether the stock appears overvalued, fairly valued, or undervalued relative to intrinsic value.

**DCF assumptions (fixed):**
- WACC: 10%
- Near-term growth rate: use `revenue_growth` from yf_data if available, else 5%
- Terminal growth rate: 3%
- Projection period: 5 years
- Base FCF: from edgar_data or yf_data `free_cash_flow`
- Shares outstanding: from yf_data `info.sharesOutstanding`

**P/E comparison:**
- Trailing P/E: `yf_data["pe_ratio"]`
- Forward P/E: `yf_data["forward_pe"]`
- Fair P/E estimate: agent derives from EPS growth rate (e.g. PEG = 1 → fair P/E ≈ EPS growth %) using `yf_data["eps"]` and `revenue_growth` as proxy; agent uses its own sector knowledge, no additional tool call
- Conclusion: whether current trailing P/E is a premium or discount to estimated fair P/E

**EV/EBITDA:**
- EV = market_cap + total_debt - cash (from yf_data info fields)
- EBITDA = from `yf_data["info"].get("ebitda")` or `edgar_data`
- If EBITDA unavailable or negative: set to `None`, mark as N/A

### Return shape

```python
{
    "summary": str,            # 2-3 sentence valuation conclusion
    "intrinsic_value": float,  # DCF per-share intrinsic value (0.0 if calculation fails)
    "upside_pct": float,       # (intrinsic_value - current_price) / current_price * 100
    "pe_analysis": str,        # e.g. "Trailing 28x vs Forward 24x — trading at a premium to estimated fair value of 22x"
    "ev_ebitda": float | None, # None if EBITDA unavailable
    "dcf_assumptions": Dict,   # {"wacc": 0.10, "growth_rate": float, "terminal_growth": 0.03, "fcf_base": float}
    "trace": List[Dict],
    "error": str | None,
}
```

### Fallback behaviour
- No API key: return dict with `intrinsic_value=0.0`, `upside_pct=0.0`, `ev_ebitda=None`, `error="No API key"`
- Missing FCF data: `intrinsic_value=0.0`, note in `summary`
- Any exception: return gracefully with `error=str(e)`, no raise

---

## 3. `run_orchestrator` Update

Add `valuation_report` parameter:

```python
def run_orchestrator(
    ticker: str,
    fundamental_report: Dict,
    catalyst_report: Dict,
    macro_report: Dict,
    valuation_report: Dict,   # NEW
    raw_data: Dict,
    api_key: str,
) -> str
```

Synthesis prompt adds:
```
VALUATION ANALYSIS:
{valuation_report["summary"] or "Not available"}
Intrinsic value: ${intrinsic_value:.2f} vs current price ${price} ({upside_pct:+.1f}% upside/downside)
```

Fallback (`_fallback()`) adds `"Valuation: {s}"` to the concatenation list.

---

## 4. `run_multi_agent_research` Changes

After the `ThreadPoolExecutor` block closes, add:

```python
if on_step:
    on_step("📊 Valuation Agent running...")
valuation_report = run_valuation_agent(ticker, yf_data, fundamental_report.get("edgar_data") or {}, api_key)
```

Pass `valuation_report` to `run_orchestrator`.

Add valuation trace entries to `trace_log`:
```python
for step in valuation_report.get("trace", []):
    step["step"] = len(trace_log)
    trace_log.append(step)
```

Add `valuation_report` to `accumulated` for `_merge_data`:
```python
accumulated = {
    ...
    "valuation": valuation_report,   # NEW
}
```

---

## 5. `_merge_data` Update (`sa_research_agent.py`)

Add one pass-through before `return merged`:

```python
merged["_valuation"] = accumulated.get("valuation") or {}
```

---

## 6. New Tab: Valuation (`stock_analyzer.py`)

### Tab order
`tab_profile, tab_tech, tab_fund, tab_valuation, tab_insider, tab_conclusion, tab_research`
(Valuation inserted as 4th tab, before Insider)

### Tab content

```
st.subheader("📊 Valuation Analysis")

[3 metric cards row]
  col1: "DCF Intrinsic Value"  → $intrinsic_value  (delta = upside_pct%)
  col2: "Current Price"        → $price
  col3: "EV/EBITDA"            → ev_ebitda or "N/A"

[P/E Comparison]
  st.markdown("#### P/E Analysis")
  st.info(pe_analysis)   ← string from agent

[DCF Assumptions expander]
  st.expander("DCF Assumptions")
    wacc, growth_rate, terminal_growth, fcf_base

[AI Summary]
  st.markdown("#### Valuation Conclusion")
  st.write(summary)
```

Show `st.warning("Valuation data unavailable")` if `_valuation` is empty or `intrinsic_value == 0.0`.

---

## 7. Files Changed

| File | Change |
|------|--------|
| `streamlit_app/sa_orchestrator.py` | Add `run_valuation_agent`; update `run_orchestrator` signature + prompt; update `run_multi_agent_research` |
| `streamlit_app/sa_research_agent.py` | Add `_valuation` pass-through in `_merge_data` (1 line) |
| `streamlit_app/stock_analyzer.py` | Add Valuation tab (7th tab); update tab list |
| `tests/test_orchestrator.py` | Add tests for `run_valuation_agent` no-key path and `run_orchestrator` with 4 reports |

---

## 8. No New Dependencies

Uses only existing `openai` package and yfinance data already in `yf_data`. No new pip packages.

---

## 9. Fallback Table

| Situation | Behaviour |
|-----------|-----------|
| No API key | `intrinsic_value=0.0`, tab shows warning |
| FCF data missing | `intrinsic_value=0.0`, summary notes missing data |
| EBITDA unavailable | `ev_ebitda=None`, tab shows "N/A" |
| Valuation agent fails | Orchestrator receives `summary=""`, tab shows warning |
| `valuation_report` missing from `run_orchestrator` call | Not applicable — only one call site (`run_multi_agent_research`), updated in same task |
