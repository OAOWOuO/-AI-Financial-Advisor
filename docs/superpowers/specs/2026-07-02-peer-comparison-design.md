# Peer Comparison — Design Spec
Date: 2026-07-02
Project: AI Financial Advisor — Stock Analyzer

---

## Overview

Add a peer comparison feature that automatically identifies well-known, analyst-covered peer companies using GPT-4o and fetches their financial data via yfinance. Results are shown in a new 8th tab ("Peers") with a color-coded comparison table. Users can also manually add extra peer tickers at any time. Peer discovery runs as part of `run_multi_agent_research`.

---

## 1. New File: `streamlit_app/sa_peers.py`

All peer logic lives in a standalone module to keep `sa_orchestrator.py` focused.

### Public interface

```python
def get_peer_tickers(ticker: str, sector: str, company_name: str, api_key: str) -> List[str]
def fetch_peer_data(tickers: List[str]) -> List[Dict]
def build_peer_table(main_ticker: str, main_data: Dict, peer_data_list: List[Dict]) -> pd.DataFrame
```

### `get_peer_tickers(ticker, sector, company_name, api_key) -> List[str]`

- **Model:** GPT-4o
- **No tools** — uses model's knowledge of financial markets
- **Prompt:** Ask for 4-5 well-known companies in the same sector/industry as `company_name` (`ticker`). Requirements: listed on a major exchange (NYSE/NASDAQ), covered by major analyst firms, commonly compared to `company_name` in equity research reports. Return only ticker symbols, one per line.
- **Output parsing:** Extract uppercase ticker symbols (2-5 chars) from response, deduplicate, exclude the main `ticker` itself.
- **Fallback (no API key or error):** Return `[]`
- **Returns:** `List[str]` of 4-5 ticker symbols

### `fetch_peer_data(tickers: List[str]) -> List[Dict]`

- Calls `_yfinance_fetch(t)` for each ticker in `tickers`
- Filters out results where `valid=False`
- **Returns:** `List[Dict]` — list of valid yfinance data dicts

### `build_peer_table(main_ticker, main_data, peer_data_list) -> pd.DataFrame`

Builds a comparison DataFrame with these 8 metrics:

| Column | Source field | Format |
|--------|-------------|--------|
| Ticker | ticker | str |
| Market Cap | market_cap | "$Xb" |
| P/E (TTM) | pe_ratio | "Xx" or "N/A" |
| Forward P/E | forward_pe | "Xx" or "N/A" |
| EV/EBITDA | info["ebitda"] + market_cap + info["totalDebt"] - info["totalCash"] | "Xx" or "N/A" |
| Revenue Growth | revenue_growth | "+X.X%" or "N/A" |
| Profit Margin | profit_margin | "X.X%" or "N/A" |
| EPS | eps | "$X.XX" or "N/A" |
| Debt/Equity | debt_to_equity | "X.Xx" or "N/A" |

Main ticker row is always first. Returns `pd.DataFrame`.

---

## 2. `run_multi_agent_research` Changes (`sa_orchestrator.py`)

After `_merge_data` call and before `return`, add:

```python
from sa_peers import get_peer_tickers, fetch_peer_data

if on_step:
    on_step("🔍 Finding peer companies...")
peer_tickers = get_peer_tickers(ticker, sector, company_name, api_key)
peer_data_list = fetch_peer_data(peer_tickers)
final_data["_peer_tickers"] = peer_tickers
final_data["_peer_data"] = peer_data_list
```

`_peer_tickers` and `_peer_data` are set directly on `final_data` (not via `_merge_data`) — same pattern used for `_orchestrator_thesis`.

---

## 3. `_merge_data` Update (`sa_research_agent.py`)

No change needed — peers are set directly on `final_data` after `_merge_data` returns, consistent with how `_orchestrator_thesis` is handled.

---

## 4. New Tab: Peers (`stock_analyzer.py`)

### Tab order (8 tabs)
`tab_profile, tab_tech, tab_fund, tab_valuation, tab_peers, tab_insider, tab_conclusion, tab_research`

### Tab content

```
st.subheader("🔍 Peer Comparison")

[If no peer data]
  st.info("No peer data available...")

[Else]
  st.dataframe(styled_table)   ← color-coded, main ticker row highlighted
  
  caption: "Source: Yahoo Finance | Peers suggested by GPT-4o"

[Manual peer input section]
  st.markdown("#### Add Custom Peers")
  col1, col2 = st.columns([3, 1])
  col1: st.text_input("Ticker symbol", key="peer_input")
  col2: st.button("Add Peer")
  
  [On button click:
    1. Call _yfinance_fetch(ticker) immediately
    2. If valid=False: show st.error(f"{ticker} not found on Yahoo Finance")
    3. If valid=True: append ticker to st.session_state["custom_peers"], store fetched data
       in st.session_state["custom_peer_data"] (dict keyed by ticker)
    4. Rebuild table combining auto peers + custom peers and re-render]
```

### Color coding (via `pd.DataFrame.style`)

- For numeric columns: cells where the peer value is **better** than main ticker shown in green, **worse** in red, within 5% shown in neutral.
- "Better" direction per metric:
  - Lower is better: P/E, Forward P/E, EV/EBITDA, Debt/Equity
  - Higher is better: Revenue Growth, Profit Margin, EPS
  - Neutral (no color): Market Cap, Ticker
- Main ticker row: always bold (via `st.markdown` table or `Styler.apply`)

### Session state keys

- `st.session_state["custom_peers"]`: `List[str]` — manually added tickers, persists across re-runs
- `st.session_state["custom_peer_data"]`: `Dict[str, Dict]` — yfinance data for each custom peer, keyed by ticker

When rendering the Peers tab, the displayed table combines `data["_peer_data"]` (auto peers from analysis run) with any custom peer data from `st.session_state["custom_peer_data"]`. `build_peer_table` is called with the merged list.

---

## 5. Files Changed

| File | Change |
|------|--------|
| `streamlit_app/sa_peers.py` | **New** — `get_peer_tickers`, `fetch_peer_data`, `build_peer_table` |
| `streamlit_app/sa_orchestrator.py` | Add peer fetch after `_merge_data`; import `sa_peers` |
| `streamlit_app/stock_analyzer.py` | Add 8th tab "Peers"; add `with tab_peers:` block |
| `tests/test_peers.py` | **New** — unit tests for `build_peer_table` and `get_peer_tickers` no-key path |

`sa_research_agent.py` — **not modified**.

---

## 6. No New Dependencies

`pandas` is already used in the existing app. No new pip packages.

---

## 7. Fallback Table

| Situation | Behaviour |
|-----------|-----------|
| No API key | `get_peer_tickers` returns `[]`; tab shows manual-input only |
| GPT-4o returns bad tickers | Filter out tickers where yfinance returns `valid=False` |
| All peer fetches fail | Table shows only main ticker row |
| Custom peer ticker invalid | Show `st.error(f"{ticker} not found")`, don't add to list |
| Peer fetch slow | Each `_yfinance_fetch` call is independent; bad ones are filtered silently |
