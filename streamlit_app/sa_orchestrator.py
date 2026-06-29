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
