import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

from sa_research_agent import (
    _yfinance_fetch,
    _yfinance_summary,
    _edgar_fetch,
    _edgar_summary,
    _web_search,
    _get_cik,
    fetch_insider_trades,
    analyze_insider_signal,
    _empty_summary,
    _merge_data,
    _run_pipeline,
    _TOOLS,
)


# ============== SUB-AGENT: FUNDAMENTAL ==============

def run_fundamental_agent(ticker: str, yf_data: Dict, api_key: str) -> Dict:
    """
    GPT-4o-mini agent: financial health via yfinance + EDGAR.
    Max 3 tool calls. Returns summary + raw edgar_data for merge.
    Falls back gracefully on any error.
    """
    trace: List[Dict] = []
    accumulated_edgar: Optional[Dict] = None
    accumulated_yf = yf_data

    if not api_key:
        return {
            "summary": "",
            "edgar_data": {},
            "yf_data": yf_data,
            "trace": trace,
            "error": "No API key",
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return {"summary": "", "edgar_data": {}, "yf_data": yf_data, "trace": trace, "error": str(e)}

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a fundamental analysis specialist. Ticker: {ticker}.\n"
                "Assess financial health using the provided data and SEC EDGAR.\n"
                "Focus on: revenue growth trend, FCF yield, operating margins, debt/equity, EPS quality.\n"
                "Call get_sec_filing with filing_type='10-K' to verify EDGAR numbers. "
                "Max 3 tool calls total. When finished, write 2-3 sentences on financial health "
                "and end your response with: FUNDAMENTAL_COMPLETE"
            ),
        },
        {
            "role": "user",
            "content": f"Yahoo Finance data:\n{_yfinance_summary(yf_data)}\n\nAnalyze {ticker} financial health.",
        },
    ]

    msg = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                max_tokens=600,
            )
        except Exception as e:
            return {"summary": "", "edgar_data": accumulated_edgar or {}, "yf_data": accumulated_yf, "trace": trace, "error": str(e)}

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
                "agent": "FundamentalAgent",
                "tool": fn,
                "args": args,
                "result_summary": "",
                "agent_reasoning": msg.content or "",
            }

            if fn == "get_yfinance_data":
                fresh = _yfinance_fetch(args.get("ticker", ticker))
                if fresh.get("valid"):
                    accumulated_yf = fresh
                result_str = _yfinance_summary(fresh)
                step["result_summary"] = f"yfinance: P/E={fresh.get('pe_ratio', 'N/A')}, FCF=${(fresh.get('free_cash_flow') or 0)/1e9:.1f}B"
            elif fn == "get_sec_filing":
                edgar = _edgar_fetch(args.get("ticker", ticker), args.get("filing_type", "10-K"))
                if edgar.get("available"):
                    accumulated_edgar = edgar
                result_str = _edgar_summary(edgar)
                step["result_summary"] = (
                    f"EDGAR: Revenue=${(edgar.get('revenue') or 0)/1e9:.1f}B, "
                    f"FCF=${(edgar.get('free_cash_flow') or 0)/1e9:.1f}B, "
                    f"EPS={edgar.get('eps', 'N/A')}"
                )
            else:
                result_str = "Tool not available in FundamentalAgent"
                step["result_summary"] = result_str

            trace.append(step)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str[:3000]})

    summary = ((msg.content if msg else None) or "").replace("FUNDAMENTAL_COMPLETE", "").strip()
    return {
        "summary": summary,
        "edgar_data": accumulated_edgar or {},
        "yf_data": accumulated_yf,
        "trace": trace,
        "error": None,
    }


# ============== SUB-AGENT: CATALYST ==============

def run_catalyst_agent(ticker: str, company_name: str, api_key: str) -> Dict:
    """
    GPT-4o-mini agent: near-term catalysts and risks from web search.
    Max 3 web_search calls. Returns summary + raw web_searches for merge.
    """
    trace: List[Dict] = []
    web_searches: List[Dict] = []

    if not api_key:
        return {"summary": "", "web_searches": web_searches, "trace": trace, "error": "No API key"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return {"summary": "", "web_searches": web_searches, "trace": trace, "error": str(e)}

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a catalyst and news analyst. Company: {company_name} ({ticker}).\n"
                "Identify near-term catalysts and risks from recent public information.\n"
                "Search for: (1) recent earnings results and management guidance, "
                "(2) analyst rating or price target changes, "
                "(3) product news, regulatory decisions, or competitive threats.\n"
                "Use web_search only. Max 3 calls. Write 2-3 sentences on key catalysts and risks. "
                "End with: CATALYST_COMPLETE"
            ),
        },
        {"role": "user", "content": f"Identify catalysts and risks for {company_name} ({ticker})."},
    ]

    msg = None
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                max_tokens=600,
            )
        except Exception as e:
            return {"summary": "", "web_searches": web_searches, "trace": trace, "error": str(e)}

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
                "agent": "CatalystAgent",
                "tool": fn,
                "args": args,
                "result_summary": "",
                "agent_reasoning": msg.content or "",
            }

            if fn == "web_search":
                query = args.get("query", "")
                result_str = _web_search(query)
                web_searches.append({"query": query, "result": result_str})
                step["result_summary"] = f'Search: "{query[:60]}" → {result_str[:100]}...'
            else:
                result_str = "Only web_search is available in CatalystAgent"
                step["result_summary"] = result_str

            trace.append(step)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str[:3000]})

    summary = ((msg.content if msg else None) or "").replace("CATALYST_COMPLETE", "").strip()
    return {"summary": summary, "web_searches": web_searches, "trace": trace, "error": None}


# ============== SUB-AGENT: MACRO ==============

def run_macro_agent(ticker: str, sector: str, api_key: str) -> Dict:
    """
    GPT-4o-mini agent: macro headwinds/tailwinds and sector positioning.
    Max 2 web_search calls.
    """
    trace: List[Dict] = []
    web_searches: List[Dict] = []

    if not api_key:
        return {"summary": "", "web_searches": web_searches, "trace": trace, "error": "No API key"}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        return {"summary": "", "web_searches": web_searches, "trace": trace, "error": str(e)}

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a macro and sector analyst. Ticker: {ticker}, Sector: {sector}.\n"
                "Assess macro headwinds/tailwinds and sector cycle positioning.\n"
                f"Search for: (1) {sector} sector outlook and competitive dynamics, "
                "(2) interest rate or inflation sensitivity for this type of business.\n"
                "Use web_search only. Max 2 calls. Write 2-3 sentences on macro and sector context. "
                "End with: MACRO_COMPLETE"
            ),
        },
        {"role": "user", "content": f"Assess macro and {sector} sector context for {ticker}."},
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
            return {"summary": "", "web_searches": web_searches, "trace": trace, "error": str(e)}

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
                "agent": "MacroAgent",
                "tool": fn,
                "args": args,
                "result_summary": "",
                "agent_reasoning": msg.content or "",
            }

            if fn == "web_search":
                query = args.get("query", "")
                result_str = _web_search(query)
                web_searches.append({"query": query, "result": result_str})
                step["result_summary"] = f'Search: "{query[:60]}" → {result_str[:100]}...'
            else:
                result_str = "Only web_search is available in MacroAgent"
                step["result_summary"] = result_str

            trace.append(step)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_str[:3000]})

    summary = ((msg.content if msg else None) or "").replace("MACRO_COMPLETE", "").strip()
    return {"summary": summary, "web_searches": web_searches, "trace": trace, "error": None}


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


# ============== ORCHESTRATOR ==============


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


# ============== ENTRY POINT ==============

def run_multi_agent_research(
    ticker: str,
    api_key: str,
    max_iterations: int = 5,
    on_step=None,
) -> Tuple[Dict, List[Dict]]:
    """
    Multi-agent stock research: three parallel sub-agents + GPT-4o orchestrator.
    Drop-in replacement for run_research_agent — identical signature and return shape.
    Routes to _run_pipeline when OpenAI is unavailable.
    """
    # Step 0: yfinance base fetch
    if on_step:
        on_step(f"Fetching Yahoo Finance data for {ticker}...")
    yf_data = _yfinance_fetch(ticker)

    fallback_trace = [{
        "step": 0,
        "agent": "Base",
        "tool": "get_yfinance_data",
        "args": {"ticker": ticker},
        "result_summary": (
            f"Price: ${yf_data.get('price', 'N/A')} | "
            f"P/E: {yf_data.get('pe_ratio', 'N/A')} | "
            f"FCF: ${(yf_data.get('free_cash_flow') or 0) / 1e9:.1f}B"
            if yf_data.get("valid")
            else f"Failed: {yf_data.get('error', 'unknown')}"
        ),
        "agent_reasoning": "Base data layer — always fetched first",
    }]

    if not yf_data.get("valid"):
        return yf_data, fallback_trace

    if not api_key:
        result, trace = _run_pipeline(ticker, yf_data, fallback_trace, on_step, reason="No OpenAI key")
        result.setdefault("_orchestrator_thesis", "")
        return result, trace

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        client.models.list()
    except Exception as e:
        result, trace = _run_pipeline(ticker, yf_data, fallback_trace, on_step, reason=str(e))
        result.setdefault("_orchestrator_thesis", "")
        return result, trace

    company_name = yf_data.get("name", ticker)
    sector = yf_data.get("sector", "Unknown")

    # Steps 1-3: parallel sub-agents
    if on_step:
        on_step("Fundamental · Catalyst · Macro agents running in parallel...")

    fundamental_report: Dict = {"summary": "", "edgar_data": {}, "yf_data": yf_data, "trace": [], "error": None}
    catalyst_report: Dict    = {"summary": "", "web_searches": [], "trace": [], "error": None}
    macro_report: Dict       = {"summary": "", "web_searches": [], "trace": [], "error": None}
    valuation_report: Dict   = {"summary": "", "intrinsic_value": 0.0, "upside_pct": 0.0,
                                 "pe_analysis": "", "ev_ebitda": None, "dcf_assumptions": {},
                                 "trace": [], "error": None}

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_fundamental_agent, ticker, yf_data, api_key): "fundamental",
            executor.submit(run_catalyst_agent, ticker, company_name, api_key): "catalyst",
            executor.submit(run_macro_agent, ticker, sector, api_key): "macro",
        }
        try:
            for future in as_completed(futures, timeout=45):
                name = futures[future]
                try:
                    result = future.result()
                    if name == "fundamental":
                        fundamental_report = result
                    elif name == "catalyst":
                        catalyst_report = result
                    else:
                        macro_report = result
                except Exception:
                    pass  # keep default empty report — orchestrator handles partial input
        except TimeoutError:
            pass  # timed-out sub-agents keep their default empty reports

    # Step 4: valuation agent (runs after fundamental to reuse edgar_data)
    if on_step:
        on_step("Valuation agent computing DCF and multiples...")
    edgar_data_for_valuation = fundamental_report.get("edgar_data") or {}
    valuation_report: Dict = run_valuation_agent(ticker, yf_data, edgar_data_for_valuation, api_key)

    if on_step:
        on_step("Orchestrator synthesizing reports...")
    thesis = run_orchestrator(ticker, fundamental_report, catalyst_report, macro_report, valuation_report, yf_data, api_key)

    # Build unified trace log
    trace_log: List[Dict] = list(fallback_trace)
    for report in (fundamental_report, catalyst_report, macro_report, valuation_report):
        for step in report.get("trace", []):
            step["step"] = len(trace_log)
            trace_log.append(step)
    trace_log.append({
        "step": len(trace_log),
        "agent": "Orchestrator",
        "tool": "synthesize",
        "args": {},
        "result_summary": (thesis[:200] + "...") if len(thesis) > 200 else thesis,
        "agent_reasoning": "GPT-4o synthesis of Fundamental + Catalyst + Macro + Valuation reports",
    })

    # Step 5: insider trades
    if on_step:
        on_step(f"Fetching SEC EDGAR Form 4 insider trades for {ticker}...")
    edgar_data = fundamental_report.get("edgar_data") or {}
    _cik = edgar_data.get("cik") or _get_cik(ticker)
    if _cik:
        insider_data = fetch_insider_trades(_cik, months=6)
        trace_log.append({
            "step": len(trace_log),
            "agent": "Base",
            "tool": "fetch_insider_trades",
            "args": {"cik": _cik, "months": 6},
            "result_summary": (
                f"Signal: {insider_data.get('summary', {}).get('signal', 'N/A')} | "
                f"{len(insider_data.get('trades', []))} trades"
            ),
            "agent_reasoning": "Post-parallel insider data fetch",
        })
    else:
        insider_data = {"available": False, "error": "No CIK", "months": 6, "trades": [], "summary": _empty_summary()}

    insider_signal_text = analyze_insider_signal(insider_data, company_name, api_key)

    accumulated = {
        "yfinance": fundamental_report.get("yf_data") or yf_data,
        "edgar": edgar_data if edgar_data.get("available") else None,
        "web_searches": catalyst_report.get("web_searches", []) + macro_report.get("web_searches", []),
        "insider": insider_data,
        "insider_signal_text": insider_signal_text,
        "orchestrator_thesis": thesis,
        "valuation": valuation_report,
    }

    final_data = _merge_data(accumulated)
    final_data["_orchestrator_thesis"] = thesis
    return final_data, trace_log
