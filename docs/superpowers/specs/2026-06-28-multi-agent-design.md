# Multi-Agent Research Architecture — Design Spec
Date: 2026-06-28
Project: AI Financial Advisor — Stock Analyzer

---

## Overview

Replace the single `run_research_agent` (one GPT-4o-mini loop doing everything) with a parallel multi-agent architecture: three specialized sub-agents run concurrently via `ThreadPoolExecutor`, then a GPT-4o Orchestrator synthesizes their reports into a coherent investment thesis. The no-LLM pipeline fallback is unchanged.

---

## 1. New File: `streamlit_app/sa_orchestrator.py`

All multi-agent logic lives here. `sa_research_agent.py` is not modified (it remains the tool/data layer).

### Public interface

```python
def run_multi_agent_research(
    ticker: str,
    api_key: str,
    on_step=None,
) -> Tuple[Dict, List[Dict]]
```

Drop-in replacement for `run_research_agent`. Same signature, same return shape. `stock_analyzer.py` changes one import line.

### Sub-agent functions

#### `run_fundamental_agent(ticker, yf_data, api_key) -> Dict`

- **Model:** GPT-4o-mini
- **Tools:** `get_yfinance_data`, `get_sec_filing` (imported from `sa_research_agent`)
- **Max tool calls:** 3
- **System prompt focus:** financial health — revenue growth, FCF yield, margins, debt/equity, EPS trend
- **Returns:**
```python
{
    "summary": str,        # 2-3 sentence financial health assessment
    "edgar_data": Dict,    # raw EDGAR result for _merge_data
    "yf_data": Dict,       # refreshed yfinance data if re-fetched
    "trace": List[Dict],   # per-step trace entries
    "error": str | None,
}
```

#### `run_catalyst_agent(ticker, company_name, api_key) -> Dict`

- **Model:** GPT-4o-mini
- **Tools:** `web_search` only
- **Max tool calls:** 3 (suggested queries: recent earnings, analyst upgrades/downgrades, product news)
- **System prompt focus:** near-term catalysts and risks from recent news, earnings surprises, guidance
- **Returns:**
```python
{
    "summary": str,          # 2-3 sentences on catalysts and risks
    "web_searches": List,    # raw search results for _merge_data
    "trace": List[Dict],
    "error": str | None,
}
```

#### `run_macro_agent(ticker, sector, api_key) -> Dict`

- **Model:** GPT-4o-mini
- **Tools:** `web_search` only
- **Max tool calls:** 2 (sector outlook, macro headwinds/tailwinds)
- **System prompt focus:** industry cycle position, interest rate sensitivity, competitive dynamics
- **Returns:**
```python
{
    "summary": str,       # 2-3 sentences on macro/sector context
    "web_searches": List,
    "trace": List[Dict],
    "error": str | None,
}
```

### Orchestrator

#### `run_orchestrator(ticker, fundamental_report, catalyst_report, macro_report, raw_data, api_key) -> str`

- **Model:** GPT-4o
- **No tools** — synthesis only
- **Input:** the three `summary` strings + key financial metrics from raw_data
- **Output:** 3–5 sentence investment thesis covering: financial quality, near-term catalysts, macro context, and a directional bias (bullish / neutral / bearish)
- **Fallback:** if GPT-4o call fails, concatenate the three summaries with headings

---

## 2. Execution Flow

```
run_multi_agent_research(ticker, api_key, on_step)
│
│  Step 0: yfinance base fetch (same as before)
│           → on_step("Fetching Yahoo Finance data...")
│
│  Steps 1-3: ThreadPoolExecutor(max_workers=3)
│   ├── FundamentalAgent  → on_step("🔢 Fundamental Agent running...")
│   ├── CatalystAgent     → on_step("📰 Catalyst Agent running...")
│   └── MacroAgent        → on_step("🌍 Macro Agent running...")
│           (all three run in parallel, timeout=45s each)
│
│  Step 4: OrchestratorAgent
│           → on_step("🧠 Orchestrator synthesizing reports...")
│
│  Step 5: fetch_insider_trades (same as before)
│           → on_step("Fetching SEC EDGAR Form 4 insider trades...")
│
└─ _merge_data(accumulated) → return (data_dict, trace_log)
```

**Threading:** `concurrent.futures.ThreadPoolExecutor` with `as_completed` and a 45-second per-agent timeout. Each sub-agent receives its own copy of data to avoid race conditions.

---

## 3. Trace Log Format

Sub-agent trace entries are prefixed with the agent name and merged into the main `trace_log`:

```python
{
    "step": int,
    "agent": "FundamentalAgent" | "CatalystAgent" | "MacroAgent" | "Orchestrator",
    "tool": str,
    "args": Dict,
    "result_summary": str,
    "agent_reasoning": str,
}
```

The Research Log tab already renders the full trace — the new `"agent"` field will be used to group entries by sub-agent (collapsible sections per agent).

---

## 4. `_merge_data` Updates

`_merge_data` in `sa_research_agent.py` receives an `accumulated` dict. The orchestrator passes:

```python
accumulated = {
    "yfinance": yf_data,
    "edgar": fundamental_report["edgar_data"],
    "web_searches": catalyst_report["web_searches"] + macro_report["web_searches"],
    "insider": insider_data,
    "insider_signal_text": insider_signal_text,
    "orchestrator_thesis": orchestrator_thesis,  # NEW key
}
```

`_merge_data` adds one new pass-through:
```python
merged["_orchestrator_thesis"] = accumulated.get("orchestrator_thesis", "")
```

The thesis is displayed in the Conclusion tab below the recommendation card (replacing the generic "trade decision" text when available).

---

## 5. `stock_analyzer.py` Changes

**Only two changes:**

1. Replace import:
```python
# Before
from sa_research_agent import run_research_agent, run_fact_checker
# After
from sa_orchestrator import run_multi_agent_research
from sa_research_agent import run_fact_checker
```

2. Replace call:
```python
# Before
_result, _trace = run_research_agent(ticker, openai_api_key, on_step=_on_step)
# After
_result, _trace = run_multi_agent_research(ticker, openai_api_key, on_step=_on_step)
```

3. **Conclusion tab:** show `_orchestrator_thesis` when available, placed AFTER the insider signal badge and BEFORE the Price Target Scenarios section:
```python
_thesis = data.get("_orchestrator_thesis", "")
if _thesis:
    st.info(_thesis)
```

4. **Research Log tab:** group trace entries by `agent` field in collapsible `st.expander` per agent.

---

## 6. Fallback Behaviour

| Situation | Behaviour |
|-----------|-----------|
| No API key | Route to existing `_run_pipeline` unchanged |
| Invalid API key (401) | Route to existing `_run_pipeline` unchanged |
| One sub-agent fails/times out | Skip it; Orchestrator receives partial reports |
| Orchestrator (GPT-4o) fails | Concatenate the three summaries as plain text |
| All three sub-agents fail | Orchestrator receives empty summaries; returns generic fallback |

---

## 7. Files Changed

| File | Change |
|------|--------|
| `streamlit_app/sa_orchestrator.py` | **New file** — all multi-agent logic |
| `streamlit_app/sa_research_agent.py` | Add `"orchestrator_thesis"` pass-through in `_merge_data` |
| `streamlit_app/stock_analyzer.py` | Replace import + call; add thesis display in Conclusion; update Research Log grouping |

`sa_research_agent.py` tool functions (`_yfinance_fetch`, `_edgar_fetch`, `_web_search`, `fetch_insider_trades`, etc.) are **not modified** — imported directly into `sa_orchestrator.py`.

---

## 8. No New Dependencies

Uses only `concurrent.futures` (stdlib) and existing `openai` package. No new pip packages.
