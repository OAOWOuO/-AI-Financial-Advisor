# Valuation Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth sub-agent (`run_valuation_agent`) that computes DCF intrinsic value, P/E comparison, and EV/EBITDA, then displays the results in a new 7th Valuation tab.

**Architecture:** `run_valuation_agent` runs serially after the three parallel sub-agents complete, reusing EDGAR data from `FundamentalAgent`. Python helper functions handle deterministic math (DCF, EV/EBITDA); the LLM interprets results and writes summaries. `run_orchestrator` gains a fourth `valuation_report` parameter. A new Valuation tab is inserted as the 4th tab in `stock_analyzer.py`.

**Tech Stack:** Python `concurrent.futures` (existing), `openai` GPT-4o-mini (existing), `streamlit` (existing). No new dependencies.

---

## File Map

| File | Change |
|------|--------|
| `streamlit_app/sa_orchestrator.py` | Add `_compute_dcf`, `_compute_ev_ebitda`, `run_valuation_agent`; update `run_orchestrator` signature and prompt; update `run_multi_agent_research` |
| `streamlit_app/sa_research_agent.py` | Add `_valuation` pass-through in `_merge_data` (1 line) |
| `streamlit_app/stock_analyzer.py` | Update tab list (6→7 tabs); add `with tab_valuation:` block; add `"ValuationAgent"` to `_agent_icons` |
| `tests/test_orchestrator.py` | Add tests for `run_valuation_agent` no-key path; update existing `run_orchestrator` tests to pass `valuation_report` |

---

## Task 1: Add DCF/EV helpers and `run_valuation_agent`

**Files:**
- Modify: `streamlit_app/sa_orchestrator.py` (append after `run_macro_agent`, before `run_orchestrator`)
- Modify: `tests/test_orchestrator.py` (append)

- [ ] **Step 1: Append tests to `tests/test_orchestrator.py`**

Add after the existing `run_macro_agent` import line at the top:
```python
from sa_orchestrator import run_valuation_agent
```

Append at the bottom of the file:
```python
_DUMMY_YF_VALUATION = {
    "valid": True, "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology",
    "price": 200.0, "pe_ratio": 29.0, "forward_pe": 25.0, "eps": 6.43,
    "free_cash_flow": 100e9, "revenue_growth": 0.08, "profit_margin": 0.25,
    "market_cap": 3e12, "debt_to_equity": 1.5, "current_ratio": 1.0,
    "info": {
        "sharesOutstanding": 15_500_000_000,
        "ebitda": 130_000_000_000,
        "totalDebt": 110_000_000_000,
        "totalCash": 165_000_000_000,
    },
}


def test_valuation_agent_no_key_returns_correct_shape():
    result = run_valuation_agent("AAPL", _DUMMY_YF_VALUATION, {}, api_key="")
    assert "summary" in result
    assert "intrinsic_value" in result
    assert "upside_pct" in result
    assert "pe_analysis" in result
    assert "ev_ebitda" in result
    assert "dcf_assumptions" in result
    assert "trace" in result
    assert "error" in result
    assert isinstance(result["trace"], list)
    assert result["error"] == "No API key"


def test_valuation_agent_no_key_computes_dcf():
    """DCF is computed in Python, not via LLM, so no API key needed."""
    result = run_valuation_agent("AAPL", _DUMMY_YF_VALUATION, {}, api_key="")
    assert result["intrinsic_value"] > 0, "DCF should compute a positive intrinsic value with valid FCF"
    assert result["upside_pct"] != 0.0


def test_valuation_agent_no_key_computes_ev_ebitda():
    result = run_valuation_agent("AAPL", _DUMMY_YF_VALUATION, {}, api_key="")
    assert result["ev_ebitda"] is not None
    assert result["ev_ebitda"] > 0


def test_valuation_agent_missing_fcf_returns_zero_intrinsic():
    yf_no_fcf = dict(_DUMMY_YF_VALUATION)
    yf_no_fcf["free_cash_flow"] = None
    yf_no_fcf["info"] = {**(_DUMMY_YF_VALUATION["info"]), "sharesOutstanding": 0}
    result = run_valuation_agent("AAPL", yf_no_fcf, {}, api_key="")
    assert result["intrinsic_value"] == 0.0
```

- [ ] **Step 2: Run to confirm ImportError**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/test_orchestrator.py::test_valuation_agent_no_key_returns_correct_shape -v 2>&1 | head -10
```

Expected: `ImportError: cannot import name 'run_valuation_agent'`

- [ ] **Step 3: Append helpers and `run_valuation_agent` to `streamlit_app/sa_orchestrator.py`**

Insert this block BETWEEN the `run_macro_agent` function and the `# ============== ORCHESTRATOR ==============` comment:

```python
# ============== VALUATION HELPERS ==============

def _compute_dcf(
    fcf_base: float,
    growth_rate: float,
    shares: float,
    wacc: float = 0.10,
    terminal_growth: float = 0.03,
    years: int = 5,
) -> Tuple[float, Dict]:
    """Returns (intrinsic_value_per_share, assumptions_dict). Returns (0.0, {}) on bad inputs."""
    if not fcf_base or fcf_base <= 0 or not shares or shares <= 0:
        return 0.0, {}
    if wacc <= terminal_growth:
        return 0.0, {}

    fcf = fcf_base
    pv_sum = 0.0
    for i in range(1, years + 1):
        fcf *= (1 + growth_rate)
        pv_sum += fcf / (1 + wacc) ** i

    terminal_value = fcf * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** years

    intrinsic_per_share = (pv_sum + pv_terminal) / shares
    return round(intrinsic_per_share, 2), {
        "wacc": wacc,
        "growth_rate": round(growth_rate, 4),
        "terminal_growth": terminal_growth,
        "fcf_base": round(fcf_base / 1e9, 2),
        "projection_years": years,
    }


def _compute_ev_ebitda(yf_data: Dict) -> Optional[float]:
    info = yf_data.get("info") or {}
    ebitda = info.get("ebitda")
    market_cap = yf_data.get("market_cap")
    total_debt = info.get("totalDebt") or 0
    cash = info.get("totalCash") or 0

    if not ebitda or ebitda <= 0 or not market_cap or market_cap <= 0:
        return None
    ev = market_cap + total_debt - cash
    if ev <= 0:
        return None
    return round(ev / ebitda, 1)


# ============== SUB-AGENT: VALUATION ==============

def run_valuation_agent(ticker: str, yf_data: Dict, edgar_data: Dict, api_key: str) -> Dict:
    """
    Computes DCF intrinsic value, P/E comparison, and EV/EBITDA for ticker.
    DCF and EV/EBITDA are computed deterministically in Python.
    GPT-4o-mini provides written P/E analysis and 2-3 sentence summary.
    Runs after FundamentalAgent to reuse its edgar_data.
    """
    trace: List[Dict] = []

    # ── Deterministic pre-computation ─────────────────────────────────────
    fcf_base = (
        (edgar_data.get("free_cash_flow") if edgar_data.get("available") else None)
        or yf_data.get("free_cash_flow")
        or 0.0
    )
    growth_rate = yf_data.get("revenue_growth") or 0.05
    shares = (yf_data.get("info") or {}).get("sharesOutstanding") or 0
    price = yf_data.get("price") or 0.0
    pe_ratio = yf_data.get("pe_ratio")
    forward_pe = yf_data.get("forward_pe")
    eps = yf_data.get("eps")

    intrinsic_value, dcf_assumptions = _compute_dcf(fcf_base, growth_rate, shares)
    upside_pct = (
        round((intrinsic_value - price) / price * 100, 1)
        if price > 0 and intrinsic_value > 0
        else 0.0
    )
    ev_ebitda = _compute_ev_ebitda(yf_data)

    _base_result: Dict = {
        "summary": "",
        "intrinsic_value": intrinsic_value,
        "upside_pct": upside_pct,
        "pe_analysis": "",
        "ev_ebitda": ev_ebitda,
        "dcf_assumptions": dcf_assumptions,
        "trace": trace,
        "error": None,
    }

    if not api_key:
        _base_result["error"] = "No API key"
        if intrinsic_value > 0:
            _base_result["summary"] = (
                f"DCF intrinsic value: ${intrinsic_value:.2f} vs current price ${price:.2f} "
                f"({upside_pct:+.1f}%). "
                f"EV/EBITDA: {f'{ev_ebitda:.1f}x' if ev_ebitda else 'N/A'}."
            )
        else:
            _base_result["summary"] = "Insufficient FCF data for DCF valuation."
        return _base_result

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        _base_result["error"] = str(e)
        return _base_result

    dcf_line = (
        f"PRE-COMPUTED DCF: ${intrinsic_value:.2f}/share "
        f"(WACC={dcf_assumptions.get('wacc', 0.10)*100:.0f}%, "
        f"growth={dcf_assumptions.get('growth_rate', 0.05)*100:.1f}%, "
        f"terminal={dcf_assumptions.get('terminal_growth', 0.03)*100:.0f}%), "
        f"implied upside/downside: {upside_pct:+.1f}%"
        if intrinsic_value > 0
        else "FCF data missing — DCF not computed. Call get_sec_filing to find FCF."
    )

    data_context = (
        f"Ticker: {ticker} | Price: ${price} | "
        f"Trailing P/E: {pe_ratio or 'N/A'} | Forward P/E: {forward_pe or 'N/A'} | EPS: ${eps or 'N/A'}\n"
        f"Revenue growth: {growth_rate*100:.1f}% | FCF base: ${fcf_base/1e9:.1f}B | "
        f"EV/EBITDA: {f'{ev_ebitda:.1f}x' if ev_ebitda else 'N/A'}\n"
        f"{dcf_line}"
    )

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a valuation specialist analyzing {ticker}.\n"
                "Tasks:\n"
                "1. If FCF is missing and DCF is not computed, call get_sec_filing (filing_type=10-K) once to find FCF.\n"
                "2. Write a P/E analysis string comparing trailing vs forward P/E and estimate fair P/E using "
                "   PEG logic (fair P/E ≈ EPS growth rate as a percentage, e.g. 15% growth → fair P/E ~15x). "
                "   Format: 'Trailing Xx vs Forward Xx — [premium/discount/inline] with estimated fair value of Xx'\n"
                "3. Write 2-3 sentences on whether the stock is overvalued / fairly valued / undervalued.\n"
                "Max 2 tool calls. Output EXACTLY this format (no other text):\n"
                "PE_ANALYSIS: [one line]\n"
                "SUMMARY: [2-3 sentences]\n"
                "VALUATION_COMPLETE"
            ),
        },
        {"role": "user", "content": data_context},
    ]

    msg = None
    for _ in range(2):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                max_tokens=500,
            )
        except Exception as e:
            _base_result["error"] = str(e)
            return _base_result

        msg = response.choices[0].message
        msg_dict: Dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        messages.append(msg_dict)

        if not msg.tool_calls:
            break

        for tc in msg.tool_calls:
            fn = tc.function.name
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}

            step = {
                "step": len(trace),
                "agent": "ValuationAgent",
                "tool": fn,
                "args": args,
                "result_summary": "",
                "agent_reasoning": msg.content or "",
            }

            if fn == "get_sec_filing":
                edgar = _edgar_fetch(args.get("ticker", ticker), args.get("filing_type", "10-K"))
                if edgar.get("available") and edgar.get("free_cash_flow"):
                    new_fcf = edgar["free_cash_flow"]
                    intrinsic_value, dcf_assumptions = _compute_dcf(new_fcf, growth_rate, shares)
                    upside_pct = (
                        round((intrinsic_value - price) / price * 100, 1)
                        if price > 0 and intrinsic_value > 0 else 0.0
                    )
                    _base_result["intrinsic_value"] = intrinsic_value
                    _base_result["upside_pct"] = upside_pct
                    _base_result["dcf_assumptions"] = dcf_assumptions
                result_str = _edgar_summary(edgar)
                step["result_summary"] = f"EDGAR FCF: ${(edgar.get('free_cash_flow') or 0)/1e9:.1f}B"
            elif fn == "get_yfinance_data":
                fresh = _yfinance_fetch(args.get("ticker", ticker))
                result_str = _yfinance_summary(fresh)
                step["result_summary"] = "yfinance refreshed"
            else:
                result_str = "Only get_sec_filing and get_yfinance_data available in ValuationAgent"
                step["result_summary"] = result_str

            trace.append(step)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str[:3000]})

    content = ((msg.content if msg else None) or "").replace("VALUATION_COMPLETE", "").strip()

    pe_analysis = ""
    summary_lines: List[str] = []
    in_summary = False
    for line in content.split("\n"):
        if line.startswith("PE_ANALYSIS:"):
            pe_analysis = line[len("PE_ANALYSIS:"):].strip()
        elif line.startswith("SUMMARY:"):
            summary_lines.append(line[len("SUMMARY:"):].strip())
            in_summary = True
        elif in_summary and line.strip():
            summary_lines.append(line.strip())

    summary = " ".join(summary_lines) if summary_lines else content

    _base_result["summary"] = summary
    _base_result["pe_analysis"] = pe_analysis
    return _base_result
```

- [ ] **Step 4: Run tests — 4 new tests should pass**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/test_orchestrator.py::test_valuation_agent_no_key_returns_correct_shape tests/test_orchestrator.py::test_valuation_agent_no_key_computes_dcf tests/test_orchestrator.py::test_valuation_agent_no_key_computes_ev_ebitda tests/test_orchestrator.py::test_valuation_agent_missing_fcf_returns_zero_intrinsic -v
```

Expected: 4 PASSED

- [ ] **Step 5: Compile check**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m py_compile streamlit_app/sa_orchestrator.py && echo "Compile OK"
```

- [ ] **Step 6: Commit**

```bash
cd /Users/oaowouo/AI-Financial-Advisor && git add streamlit_app/sa_orchestrator.py tests/test_orchestrator.py && git commit -m "feat: add run_valuation_agent with DCF, P/E, and EV/EBITDA computation"
```

---

## Task 2: Update `run_orchestrator` to include valuation report

**Files:**
- Modify: `streamlit_app/sa_orchestrator.py` — replace `run_orchestrator` function
- Modify: `tests/test_orchestrator.py` — update existing orchestrator tests

- [ ] **Step 1: Update existing orchestrator tests in `tests/test_orchestrator.py`**

The two existing tests call `run_orchestrator` with 6 args. They need a 7th arg (`valuation_report`). Find and replace both test functions:

Replace `test_orchestrator_no_key_concatenates_summaries`:
```python
def test_orchestrator_no_key_concatenates_summaries():
    fundamental = {"summary": "Strong FCF of $100B.", "edgar_data": {}, "yf_data": {}, "trace": [], "error": None}
    catalyst    = {"summary": "Earnings beat by 5%.", "web_searches": [], "trace": [], "error": None}
    macro       = {"summary": "Tech sector favourable.", "web_searches": [], "trace": [], "error": None}
    valuation   = {"summary": "DCF shows 15% upside.", "intrinsic_value": 230.0, "upside_pct": 15.0,
                   "pe_analysis": "", "ev_ebitda": 22.5, "dcf_assumptions": {}, "trace": [], "error": None}
    raw = {"price": 200.0, "market_cap": 3e12}
    result = run_orchestrator("AAPL", fundamental, catalyst, macro, valuation, raw, api_key="")
    assert "Strong FCF" in result
    assert "Earnings beat" in result
    assert "Tech sector" in result
    assert "DCF shows" in result
```

Replace `test_orchestrator_empty_reports_returns_empty_string`:
```python
def test_orchestrator_empty_reports_returns_empty_string():
    fundamental = {"summary": "", "edgar_data": {}, "yf_data": {}, "trace": [], "error": "No API key"}
    catalyst    = {"summary": "", "web_searches": [], "trace": [], "error": "No API key"}
    macro       = {"summary": "", "web_searches": [], "trace": [], "error": "No API key"}
    valuation   = {"summary": "", "intrinsic_value": 0.0, "upside_pct": 0.0,
                   "pe_analysis": "", "ev_ebitda": None, "dcf_assumptions": {}, "trace": [], "error": "No API key"}
    raw = {"price": 100.0, "market_cap": 1e12}
    result = run_orchestrator("XYZ", fundamental, catalyst, macro, valuation, raw, api_key="")
    assert isinstance(result, str)
```

- [ ] **Step 2: Run to confirm tests fail (wrong arg count)**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/test_orchestrator.py::test_orchestrator_no_key_concatenates_summaries -v 2>&1 | head -15
```

Expected: `TypeError` (wrong number of arguments)

- [ ] **Step 3: Replace `run_orchestrator` in `streamlit_app/sa_orchestrator.py`**

Find the entire `run_orchestrator` function (from `def run_orchestrator(` to the closing `return _fallback()`) and replace with:

```python
def run_orchestrator(
    ticker: str,
    fundamental_report: Dict,
    catalyst_report: Dict,
    macro_report: Dict,
    valuation_report: Dict,
    raw_data: Dict,
    api_key: str,
) -> str:
    """
    GPT-4o synthesis of four sub-agent reports into a 3-5 sentence investment thesis.
    Falls back to deterministic concatenation on any failure or missing API key.
    """
    def _fallback() -> str:
        parts = []
        for label, report in [
            ("Fundamentals", fundamental_report),
            ("Catalysts", catalyst_report),
            ("Macro", macro_report),
            ("Valuation", valuation_report),
        ]:
            s = report.get("summary", "").strip()
            if s:
                parts.append(f"{label}: {s}")
        return " ".join(parts)

    if not api_key:
        return _fallback()

    price = raw_data.get("price", "N/A")
    mkt_cap = round((raw_data.get("market_cap") or 0) / 1e9, 1)
    intrinsic_value = valuation_report.get("intrinsic_value", 0.0)
    upside_pct = valuation_report.get("upside_pct", 0.0)

    valuation_line = (
        f"DCF intrinsic value: ${intrinsic_value:.2f} ({upside_pct:+.1f}% upside/downside)\n"
        if intrinsic_value > 0
        else ""
    )

    prompt = (
        f"You are a senior portfolio manager. Synthesize four research reports on {ticker} "
        f"(current price: ${price}, market cap: ${mkt_cap}B) into a coherent 3-5 sentence investment thesis.\n\n"
        f"FUNDAMENTAL ANALYSIS:\n{fundamental_report.get('summary') or 'Not available'}\n\n"
        f"CATALYST ANALYSIS:\n{catalyst_report.get('summary') or 'Not available'}\n\n"
        f"MACRO/SECTOR ANALYSIS:\n{macro_report.get('summary') or 'Not available'}\n\n"
        f"VALUATION ANALYSIS:\n{valuation_report.get('summary') or 'Not available'}\n"
        f"{valuation_line}\n"
        "Write a balanced, specific thesis covering: financial quality, near-term catalysts, "
        "macro context, valuation, and a clear directional bias (bullish / neutral / bearish)."
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return _fallback()
```

- [ ] **Step 4: Run all orchestrator tests — all should pass**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/test_orchestrator.py -v 2>&1 | tail -15
```

Expected: 11 PASSED (7 original + 4 new)

- [ ] **Step 5: Compile check**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m py_compile streamlit_app/sa_orchestrator.py && echo "Compile OK"
```

- [ ] **Step 6: Commit**

```bash
cd /Users/oaowouo/AI-Financial-Advisor && git add streamlit_app/sa_orchestrator.py tests/test_orchestrator.py && git commit -m "feat: update run_orchestrator to include valuation_report (4-agent synthesis)"
```

---

## Task 3: Update `run_multi_agent_research` to call valuation agent

**Files:**
- Modify: `streamlit_app/sa_orchestrator.py` — 4 targeted edits in `run_multi_agent_research`

The current `run_multi_agent_research` is in `sa_orchestrator.py`. Make these changes:

- [ ] **Step 1: Add `valuation_report` default before the ThreadPoolExecutor block**

Find this block (around line 436):
```python
    fundamental_report: Dict = {"summary": "", "edgar_data": {}, "yf_data": yf_data, "trace": [], "error": None}
    catalyst_report: Dict    = {"summary": "", "web_searches": [], "trace": [], "error": None}
    macro_report: Dict       = {"summary": "", "web_searches": [], "trace": [], "error": None}
```

Replace with:
```python
    fundamental_report: Dict = {"summary": "", "edgar_data": {}, "yf_data": yf_data, "trace": [], "error": None}
    catalyst_report: Dict    = {"summary": "", "web_searches": [], "trace": [], "error": None}
    macro_report: Dict       = {"summary": "", "web_searches": [], "trace": [], "error": None}
    valuation_report: Dict   = {"summary": "", "intrinsic_value": 0.0, "upside_pct": 0.0,
                                 "pe_analysis": "", "ev_ebitda": None, "dcf_assumptions": {},
                                 "trace": [], "error": None}
```

- [ ] **Step 2: Add valuation agent call after the ThreadPoolExecutor block**

Find this line (the `on_step("🧠 Orchestrator...")` line, around line 462):
```python
    if on_step:
        on_step("🧠 Orchestrator synthesizing reports...")
    thesis = run_orchestrator(ticker, fundamental_report, catalyst_report, macro_report, yf_data, api_key)
```

Replace with:
```python
    if on_step:
        on_step("📊 Valuation Agent running...")
    valuation_report = run_valuation_agent(
        ticker, yf_data, fundamental_report.get("edgar_data") or {}, api_key
    )

    if on_step:
        on_step("🧠 Orchestrator synthesizing reports...")
    thesis = run_orchestrator(ticker, fundamental_report, catalyst_report, macro_report, valuation_report, yf_data, api_key)
```

- [ ] **Step 3: Add valuation trace entries to the trace merge loop**

Find:
```python
    for report in (fundamental_report, catalyst_report, macro_report):
        for step in report.get("trace", []):
            step["step"] = len(trace_log)
            trace_log.append(step)
```

Replace with:
```python
    for report in (fundamental_report, catalyst_report, macro_report, valuation_report):
        for step in report.get("trace", []):
            step["step"] = len(trace_log)
            trace_log.append(step)
```

- [ ] **Step 4: Add `valuation` key to `accumulated` dict**

Find (near the bottom of `run_multi_agent_research`):
```python
    accumulated = {
        "yfinance": fundamental_report.get("yf_data") or yf_data,
        "edgar": edgar_data if edgar_data.get("available") else None,
        "web_searches": catalyst_report.get("web_searches", []) + macro_report.get("web_searches", []),
        "insider": insider_data,
        "insider_signal_text": insider_signal_text,
        "orchestrator_thesis": thesis,
    }
```

Replace with:
```python
    accumulated = {
        "yfinance": fundamental_report.get("yf_data") or yf_data,
        "edgar": edgar_data if edgar_data.get("available") else None,
        "web_searches": catalyst_report.get("web_searches", []) + macro_report.get("web_searches", []),
        "insider": insider_data,
        "insider_signal_text": insider_signal_text,
        "orchestrator_thesis": thesis,
        "valuation": valuation_report,
    }
```

- [ ] **Step 5: Compile check**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m py_compile streamlit_app/sa_orchestrator.py && echo "Compile OK"
```

- [ ] **Step 6: Run all tests — 11 should still pass**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/test_orchestrator.py -v 2>&1 | tail -10
```

Expected: 11 PASSED

- [ ] **Step 7: Commit**

```bash
cd /Users/oaowouo/AI-Financial-Advisor && git add streamlit_app/sa_orchestrator.py && git commit -m "feat: wire run_valuation_agent into run_multi_agent_research"
```

---

## Task 4: Add `_valuation` pass-through in `_merge_data`

**Files:**
- Modify: `streamlit_app/sa_research_agent.py` — 1 line insertion

- [ ] **Step 1: Find and edit `_merge_data` return block**

In `sa_research_agent.py`, find these lines near the end of `_merge_data` (around line 639):
```python
    merged["_insider_signal_text"] = accumulated.get("insider_signal_text") or ""
    merged["_orchestrator_thesis"] = accumulated.get("orchestrator_thesis") or ""

    return merged
```

Insert one line before `return merged`:
```python
    merged["_insider_signal_text"] = accumulated.get("insider_signal_text") or ""
    merged["_orchestrator_thesis"] = accumulated.get("orchestrator_thesis") or ""
    merged["_valuation"] = accumulated.get("valuation") or {}

    return merged
```

- [ ] **Step 2: Compile check**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m py_compile streamlit_app/sa_research_agent.py && echo "Compile OK"
```

- [ ] **Step 3: Run full test suite**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/test_insider_trades.py tests/test_orchestrator.py -v 2>&1 | tail -8
```

Expected: 29 PASSED (18 insider + 11 orchestrator)

- [ ] **Step 4: Commit**

```bash
cd /Users/oaowouo/AI-Financial-Advisor && git add streamlit_app/sa_research_agent.py && git commit -m "feat: add _valuation pass-through in _merge_data"
```

---

## Task 5: Add Valuation tab to `stock_analyzer.py`

**Files:**
- Modify: `streamlit_app/stock_analyzer.py` — 3 targeted edits

- [ ] **Step 1: Update tab list (line 1997)**

Find:
```python
        tab_profile, tab_tech, tab_fund, tab_insider, tab_conclusion, tab_research = st.tabs(
            ["🏢 Profile", "📊 Technical Analysis", "📋 Fundamental Analysis", "👥 Insider", "🎯 Conclusion & Forecast", "🔍 Research Log"]
        )
```

Replace with:
```python
        tab_profile, tab_tech, tab_fund, tab_valuation, tab_insider, tab_conclusion, tab_research = st.tabs(
            ["🏢 Profile", "📊 Technical Analysis", "📋 Fundamental Analysis", "💰 Valuation", "👥 Insider", "🎯 Conclusion & Forecast", "🔍 Research Log"]
        )
```

- [ ] **Step 2: Insert `with tab_valuation:` block**

Find the line `        with tab_insider:` (around line 2735). Insert the following block BEFORE it:

```python
        # ── VALUATION TAB ──────────────────────────────────────────────────────
        with tab_valuation:
            if not has_data:
                st.info("Run an analysis to see valuation estimates.")
            else:
                _val = data.get("_valuation") or {}
                _iv = _val.get("intrinsic_value", 0.0)
                _upside = _val.get("upside_pct", 0.0)
                _ev_ebitda = _val.get("ev_ebitda")
                _pe_analysis = _val.get("pe_analysis", "")
                _val_summary = _val.get("summary", "")
                _dcf_assump = _val.get("dcf_assumptions") or {}
                _cur_price = data.get("price") or 0.0

                if not _val or _iv == 0.0:
                    st.warning(
                        "Valuation data unavailable. "
                        "This may be due to missing FCF data or no OpenAI API key."
                    )
                else:
                    st.subheader("📊 Valuation Analysis")

                    _vcol1, _vcol2, _vcol3 = st.columns(3)
                    with _vcol1:
                        st.metric(
                            "DCF Intrinsic Value",
                            f"${_iv:.2f}",
                            delta=f"{_upside:+.1f}% upside" if _upside >= 0 else f"{_upside:.1f}% downside",
                            delta_color="normal",
                        )
                    with _vcol2:
                        st.metric("Current Price", f"${_cur_price:.2f}")
                    with _vcol3:
                        st.metric(
                            "EV/EBITDA",
                            f"{_ev_ebitda:.1f}x" if _ev_ebitda is not None else "N/A",
                        )

                    if _pe_analysis:
                        st.markdown("#### P/E Analysis")
                        st.info(_pe_analysis)

                    if _dcf_assump:
                        with st.expander("DCF Assumptions"):
                            st.write(f"- **WACC:** {_dcf_assump.get('wacc', 0.10)*100:.0f}%")
                            st.write(f"- **Near-term growth:** {_dcf_assump.get('growth_rate', 0.05)*100:.1f}%")
                            st.write(f"- **Terminal growth:** {_dcf_assump.get('terminal_growth', 0.03)*100:.0f}%")
                            st.write(f"- **Base FCF:** ${_dcf_assump.get('fcf_base', 0):.2f}B")
                            st.write(f"- **Projection period:** {_dcf_assump.get('projection_years', 5)} years")

                    if _val_summary:
                        st.markdown("#### Valuation Conclusion")
                        st.write(_val_summary)

```

- [ ] **Step 3: Add `"ValuationAgent"` to `_agent_icons` dict in Research Log**

Find in the Research Log tab (around line 3070):
```python
                    _agent_icons = {
                        "FundamentalAgent": "🔢",
                        "CatalystAgent": "📰",
                        "MacroAgent": "🌍",
                        "Orchestrator": "🧠",
                        "Base": "⚙️",
                    }
```

Replace with:
```python
                    _agent_icons = {
                        "FundamentalAgent": "🔢",
                        "CatalystAgent": "📰",
                        "MacroAgent": "🌍",
                        "ValuationAgent": "💰",
                        "Orchestrator": "🧠",
                        "Base": "⚙️",
                    }
```

- [ ] **Step 4: Compile check**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m py_compile streamlit_app/stock_analyzer.py && echo "Compile OK"
```

- [ ] **Step 5: Run full test suite**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/test_insider_trades.py tests/test_orchestrator.py -v 2>&1 | tail -8
```

Expected: 29 PASSED

- [ ] **Step 6: Commit**

```bash
cd /Users/oaowouo/AI-Financial-Advisor && git add streamlit_app/stock_analyzer.py && git commit -m "feat: add Valuation tab with DCF intrinsic value, P/E analysis, EV/EBITDA"
```

---

## Task 6: Final CI check

**Files:** none — verification only

- [ ] **Step 1: Compile all modified files**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m py_compile streamlit_app/sa_orchestrator.py streamlit_app/sa_research_agent.py streamlit_app/stock_analyzer.py && echo "All compile OK"
```

Expected: `All compile OK`

- [ ] **Step 2: Run full test suite**

```
cd /Users/oaowouo/AI-Financial-Advisor && python -m pytest tests/test_insider_trades.py tests/test_orchestrator.py -v 2>&1 | tail -10
```

Expected: 29 PASSED

- [ ] **Step 3: Push**

```bash
git push origin main
```

---

## Self-Review Checklist

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `run_valuation_agent(ticker, yf_data, edgar_data, api_key)` signature | Task 1 |
| GPT-4o-mini model | Task 1 |
| Max 2 tool calls | Task 1 |
| DCF: WACC 10%, terminal 3%, 5yr, uses FCF + shares | Task 1 (`_compute_dcf`) |
| P/E: trailing vs forward vs fair P/E estimate | Task 1 (agent prompt) |
| EV/EBITDA: market_cap + debt - cash / EBITDA | Task 1 (`_compute_ev_ebitda`) |
| Returns 8-key dict with all specified fields | Task 1 |
| Graceful fallback on no key / missing FCF | Task 1 |
| `run_orchestrator` gains `valuation_report` param | Task 2 |
| Fallback `_fallback()` includes Valuation | Task 2 |
| `run_multi_agent_research` calls valuation agent after parallel block | Task 3 |
| Valuation trace entries merged into `trace_log` | Task 3 |
| `accumulated["valuation"]` passed to `_merge_data` | Task 3 |
| `_merge_data` passes `_valuation` through | Task 4 |
| Valuation tab (7th tab) with 3 metrics + P/E + DCF expander + summary | Task 5 |
| Tab order: Profile, Technical, Fundamental, **Valuation**, Insider, Conclusion, Research | Task 5 |
| `st.warning` when `_iv == 0.0` | Task 5 |
| `"ValuationAgent": "💰"` in `_agent_icons` | Task 5 |

All spec requirements covered. No TBDs. No placeholders.
