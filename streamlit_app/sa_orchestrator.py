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


# ============== ORCHESTRATOR ==============


def run_orchestrator(
    ticker: str,
    fundamental_report: Dict,
    catalyst_report: Dict,
    macro_report: Dict,
    raw_data: Dict,
    api_key: str,
) -> str:
    """
    GPT-4o synthesis of three sub-agent reports into a 3-5 sentence investment thesis.
    Falls back to deterministic concatenation on any failure or missing API key.
    """
    def _fallback() -> str:
        parts = []
        for label, report in [
            ("Fundamentals", fundamental_report),
            ("Catalysts", catalyst_report),
            ("Macro", macro_report),
        ]:
            s = report.get("summary", "").strip()
            if s:
                parts.append(f"{label}: {s}")
        return " ".join(parts)

    if not api_key:
        return _fallback()

    price = raw_data.get("price", "N/A")
    mkt_cap = round((raw_data.get("market_cap") or 0) / 1e9, 1)

    prompt = (
        f"You are a senior portfolio manager. Synthesize three research reports on {ticker} "
        f"(current price: ${price}, market cap: ${mkt_cap}B) into a coherent 3-5 sentence investment thesis.\n\n"
        f"FUNDAMENTAL ANALYSIS:\n{fundamental_report.get('summary') or 'Not available'}\n\n"
        f"CATALYST ANALYSIS:\n{catalyst_report.get('summary') or 'Not available'}\n\n"
        f"MACRO/SECTOR ANALYSIS:\n{macro_report.get('summary') or 'Not available'}\n\n"
        "Write a balanced, specific thesis covering: financial quality, near-term catalysts, "
        "macro context, and a clear directional bias (bullish / neutral / bearish)."
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
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

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_fundamental_agent, ticker, yf_data, api_key): "fundamental",
            executor.submit(run_catalyst_agent, ticker, company_name, api_key): "catalyst",
            executor.submit(run_macro_agent, ticker, sector, api_key): "macro",
        }
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

    if on_step:
        on_step("Orchestrator synthesizing reports...")
    thesis = run_orchestrator(ticker, fundamental_report, catalyst_report, macro_report, yf_data, api_key)

    # Build unified trace log
    trace_log: List[Dict] = list(fallback_trace)
    for report in (fundamental_report, catalyst_report, macro_report):
        for step in report.get("trace", []):
            step["step"] = len(trace_log)
            trace_log.append(step)
    trace_log.append({
        "step": len(trace_log),
        "agent": "Orchestrator",
        "tool": "synthesize",
        "args": {},
        "result_summary": (thesis[:200] + "...") if len(thesis) > 200 else thesis,
        "agent_reasoning": "GPT-4o synthesis of Fundamental + Catalyst + Macro reports",
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
    }

    final_data = _merge_data(accumulated)
    final_data["_orchestrator_thesis"] = thesis
    return final_data, trace_log
