import json
import re
import time
import xml.etree.ElementTree as ET
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

_EDGAR_HEADERS = {
    "User-Agent": "AI-Financial-Advisor contact@example.com",
    "Accept-Encoding": "gzip, deflate",
}


# ============== TOOL: YFINANCE ==============

def _yfinance_fetch(ticker: str) -> Dict:
    """Fetch comprehensive stock data from Yahoo Finance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        hist_1y = stock.history(period="1y")
        hist_2y = stock.history(period="2y")
        hist_5y = stock.history(period="5y")

        if hist_1y.empty:
            return {"valid": False, "ticker": ticker, "error": "No price history found"}

        return {
            "valid": True,
            "ticker": ticker,
            "info": info,
            "hist_1y": hist_1y,
            "hist_2y": hist_2y,
            "hist_5y": hist_5y,
            "income_stmt": stock.income_stmt,
            "balance_sheet": stock.balance_sheet,
            "cash_flow": stock.cashflow,
            "quarterly_income": stock.quarterly_income_stmt,
            "quarterly_bs": stock.quarterly_balance_sheet,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice") or float(hist_1y["Close"].iloc[-1]),
            "prev_close": info.get("previousClose", 0),
            "open": info.get("open", 0),
            "high": info.get("dayHigh", 0),
            "low": info.get("dayLow", 0),
            "volume": info.get("volume", 0),
            "avg_volume": info.get("averageVolume", 0),
            "avg_volume_10d": info.get("averageVolume10days", 0),
            "market_cap": info.get("marketCap", 0),
            "enterprise_value": info.get("enterpriseValue", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "float_shares": info.get("floatShares", 0),
            "high_52w": info.get("fiftyTwoWeekHigh", 0),
            "low_52w": info.get("fiftyTwoWeekLow", 0),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "ev_revenue": info.get("enterpriseToRevenue"),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "free_cash_flow": info.get("freeCashflow"),
            "operating_cash_flow": info.get("operatingCashflow"),
            "eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "book_value": info.get("bookValue"),
            "revenue_per_share": info.get("revenuePerShare"),
            "dividend_yield": (lambda v: v if 0 < v <= 0.15 else None)(info.get("dividendYield") or 0),
            "dividend_rate": info.get("dividendRate", 0),
            "payout_ratio": info.get("payoutRatio"),
            "target_price": info.get("targetMeanPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "analyst_rating": info.get("recommendationKey"),
            "num_analysts": info.get("numberOfAnalystOpinions", 0),
            "beta": info.get("beta", 1),
        }
    except Exception as e:
        return {"valid": False, "ticker": ticker, "error": str(e)}


def _yfinance_summary(data: Dict) -> str:
    """Compact JSON summary of yfinance data for agent context window."""
    if not data.get("valid"):
        return f"yfinance failed: {data.get('error', 'unknown error')}"
    return json.dumps({
        "name": data.get("name"),
        "price": data.get("price"),
        "market_cap_B": round((data.get("market_cap") or 0) / 1e9, 2),
        "sector": data.get("sector"),
        "industry": data.get("industry"),
        "pe_ratio": data.get("pe_ratio"),
        "forward_pe": data.get("forward_pe"),
        "peg_ratio": data.get("peg_ratio"),
        "eps": data.get("eps"),
        "forward_eps": data.get("forward_eps"),
        "revenue_growth": data.get("revenue_growth"),
        "earnings_growth": data.get("earnings_growth"),
        "profit_margin": data.get("profit_margin"),
        "roe": data.get("roe"),
        "debt_to_equity": data.get("debt_to_equity"),
        "current_ratio": data.get("current_ratio"),
        "free_cash_flow_B": round((data.get("free_cash_flow") or 0) / 1e9, 2),
        "analyst_rating": data.get("analyst_rating"),
        "target_price": data.get("target_price"),
        "num_analysts": data.get("num_analysts"),
        "beta": data.get("beta"),
    }, default=str)


# ============== TOOL: SEC EDGAR ==============

def _get_cik(ticker: str) -> Optional[str]:
    """Look up SEC CIK for a ticker via SEC's company_tickers.json."""
    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=_EDGAR_HEADERS,
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        ticker_upper = ticker.upper()
        for entry in resp.json().values():
            if entry.get("ticker", "").upper() == ticker_upper:
                return str(entry["cik_str"]).zfill(10)
        return None
    except Exception:
        return None


def _get_most_recent_annual(units_list: List[Dict]) -> Optional[float]:
    """Return the most recent 10-K (or 20-F) value from an EDGAR units list."""
    annual = [u for u in units_list if u.get("form") in ("10-K", "20-F", "10-K/A")]
    if not annual:
        return None
    annual.sort(key=lambda x: x.get("end", ""), reverse=True)
    return annual[0].get("val")


def _fetch_concept(cik: str, concept: str, taxonomy: str = "us-gaap") -> Optional[float]:
    """Fetch most recent annual value for one XBRL concept from EDGAR."""
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/{taxonomy}/{concept}.json"
    try:
        resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=8)
        if resp.status_code != 200:
            return None
        units = resp.json().get("units", {})
        # Try all known unit types: USD, shares, USD/shares (used for EPS)
        for unit_type in ("USD", "shares", "USD/shares"):
            vals = units.get(unit_type, [])
            if vals:
                v = _get_most_recent_annual(vals)
                if v is not None:
                    return v
        return None
    except Exception:
        return None


def _edgar_fetch(ticker: str, filing_type: str = "10-K") -> Dict:
    """
    Fetch verified financial data from SEC EDGAR using per-concept XBRL API.
    Falls back gracefully: returns {"available": False} if ticker not found.
    """
    cik = _get_cik(ticker)
    if not cik:
        return {"available": False, "error": f"CIK not found for {ticker} — not listed on SEC EDGAR (may be non-US)"}

    result = {"available": True, "ticker": ticker, "cik": cik, "filing_type": filing_type}

    # Revenue — try multiple GAAP tags in order of preference
    revenue = None
    for tag in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"):
        revenue = _fetch_concept(cik, tag)
        if revenue is not None:
            break
        time.sleep(0.1)

    result["revenue"] = revenue

    # Net income
    time.sleep(0.1)
    result["net_income"] = _fetch_concept(cik, "NetIncomeLoss")

    # EPS diluted (fall back to basic)
    time.sleep(0.1)
    eps = _fetch_concept(cik, "EarningsPerShareDiluted")
    if eps is None:
        time.sleep(0.1)
        eps = _fetch_concept(cik, "EarningsPerShareBasic")
    result["eps"] = eps

    # Cash flows
    time.sleep(0.1)
    op_cf = _fetch_concept(cik, "NetCashProvidedByUsedInOperatingActivities")
    result["operating_cash_flow"] = op_cf

    time.sleep(0.1)
    capex = _fetch_concept(cik, "PaymentsToAcquirePropertyPlantAndEquipment")
    result["capex"] = capex
    result["free_cash_flow"] = (op_cf - capex) if (op_cf and capex) else None

    # Balance sheet
    time.sleep(0.1)
    result["total_debt"] = _fetch_concept(cik, "LongTermDebt") or _fetch_concept(cik, "LongTermDebtNoncurrent")

    time.sleep(0.1)
    result["shares_outstanding"] = _fetch_concept(cik, "CommonStockSharesOutstanding")

    # Remove None values for a clean dict
    return {k: v for k, v in result.items() if v is not None}


def _edgar_summary(edgar: Dict) -> str:
    """Compact summary of EDGAR data for agent context."""
    if not edgar.get("available"):
        return f"EDGAR: {edgar.get('error', 'unavailable')}"
    return json.dumps({
        "source": "SEC EDGAR (verified)",
        "revenue_B": round((edgar.get("revenue") or 0) / 1e9, 2),
        "net_income_B": round((edgar.get("net_income") or 0) / 1e9, 2),
        "eps": edgar.get("eps"),
        "operating_cash_flow_B": round((edgar.get("operating_cash_flow") or 0) / 1e9, 2),
        "free_cash_flow_B": round((edgar.get("free_cash_flow") or 0) / 1e9, 2),
        "total_debt_B": round((edgar.get("total_debt") or 0) / 1e9, 2),
        "shares_outstanding_M": round((edgar.get("shares_outstanding") or 0) / 1e6, 1),
    }, default=str)


# ============== TOOL: WEB SEARCH ==============

def _web_search(query: str, max_results: int = 4) -> str:
    """DuckDuckGo text search. Returns formatted text snippets."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            return "No results found."
        return "\n".join(f"[{r.get('title', '')}] {r.get('body', '')}" for r in results)[:3000]
    except ImportError:
        return "Web search unavailable: install ddgs."
    except Exception as e:
        return f"Search error: {e}"


# ============== OPENAI TOOL SCHEMAS ==============

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_yfinance_data",
            "description": "Fetch price, valuation ratios, growth metrics, and analyst targets from Yahoo Finance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol e.g. AAPL"}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sec_filing",
            "description": (
                "Fetch verified financial data (revenue, EPS, FCF, debt) directly from SEC EDGAR XBRL filings. "
                "More authoritative than Yahoo Finance for exact financial statement numbers. "
                "Only works for US-listed companies that file with the SEC."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "filing_type": {
                        "type": "string",
                        "enum": ["10-K", "10-Q"],
                        "description": "10-K for annual, 10-Q for quarterly",
                    },
                },
                "required": ["ticker", "filing_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent news, competitor analysis, and industry trends about a company.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string"}
                },
                "required": ["query"],
            },
        },
    },
]


# ============== DETERMINISTIC PIPELINE (no-LLM fallback) ==============

def _run_pipeline(
    ticker: str,
    yf_data: Dict,
    trace_log: List[Dict],
    on_step,
    reason: str = "",
) -> Tuple[Dict, List[Dict]]:
    """
    No-LLM mode: always runs yfinance → EDGAR → web search in fixed order.
    Used when OpenAI is unavailable or the API key is invalid.
    """
    trace_log = list(trace_log)
    trace_log[0]["agent_reasoning"] = f"Pipeline mode (no LLM): {reason}"

    accumulated = {"yfinance": yf_data, "edgar": None, "web_searches": []}

    # Step 1: EDGAR
    if on_step:
        on_step(f"Step 1: fetching SEC EDGAR 10-K for {ticker}...")
    edgar = _edgar_fetch(ticker)
    if edgar.get("available"):
        accumulated["edgar"] = edgar
        summary = (
            f"Revenue: ${(edgar.get('revenue') or 0)/1e9:.1f}B | "
            f"FCF: ${(edgar.get('free_cash_flow') or 0)/1e9:.1f}B | "
            f"EPS: {edgar.get('eps', 'N/A')}"
        )
    else:
        summary = f"EDGAR unavailable — {edgar.get('error', 'unknown')}"
    trace_log.append({
        "step": 1,
        "tool": "get_sec_filing",
        "args": {"ticker": ticker, "filing_type": "10-K"},
        "result_summary": summary,
        "agent_reasoning": "Pipeline step (fixed order, no LLM)",
    })

    # Step 2: web search
    if on_step:
        on_step(f"Step 2: web search for {ticker} news & industry...")
    query = f"{ticker} stock news outlook competitors 2025"
    text = _web_search(query)
    accumulated["web_searches"].append({"query": query, "result": text})
    trace_log.append({
        "step": 2,
        "tool": "web_search",
        "args": {"query": query},
        "result_summary": text[:150] + "...",
        "agent_reasoning": "Pipeline step (fixed order, no LLM)",
    })

    return _merge_data(accumulated), trace_log


# ============== RESEARCH AGENT ==============

def run_research_agent(
    ticker: str,
    api_key: str,
    max_iterations: int = 5,
    on_step=None,
) -> Tuple[Dict, List[Dict]]:
    """
    Agent-driven stock research.

    Returns (data_dict, trace_log).
    data_dict is compatible with calculate_valuation / generate_recommendation.
    Falls back to pure yfinance if OpenAI is unavailable.

    on_step: optional callback(str) called with a status message after each tool call.
    """
    # Step 0: always fetch yfinance first — it's the base layer and the fallback
    if on_step:
        on_step(f"Fetching Yahoo Finance data for {ticker}...")
    yf_data = _yfinance_fetch(ticker)

    fallback_trace = [{
        "step": 0,
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
        return _run_pipeline(ticker, yf_data, fallback_trace, on_step, reason="No OpenAI key")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Quick connectivity check — catches 401 / wrong key before the loop
        client.models.list()
    except Exception as e:
        return _run_pipeline(ticker, yf_data, fallback_trace, on_step, reason=str(e))

    accumulated = {"yfinance": yf_data, "edgar": None, "web_searches": []}
    trace_log = list(fallback_trace)

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a stock research agent. Ticker: {ticker}.\n\n"
                f"Yahoo Finance data was already fetched (see step 0 summary below). "
                f"You can call up to {max_iterations - 1} more tools to fill data gaps or verify numbers.\n\n"
                "Recommended strategy:\n"
                "1. Call get_sec_filing('10-K') to verify EPS, FCF, and revenue with official SEC data\n"
                "2. Call web_search for recent news or competitive context if useful\n"
                "3. Stop when data is sufficient for a complete institutional analysis\n\n"
                "Be selective. Avoid redundant calls. When done, reply with just: RESEARCH_COMPLETE"
            ),
        },
        {
            "role": "user",
            "content": f"Yahoo Finance summary for {ticker}:\n{_yfinance_summary(yf_data)}\n\nProceed.",
        },
    ]

    for iteration in range(max_iterations - 1):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                max_tokens=800,
            )
        except Exception as e:
            trace_log.append({
                "step": len(trace_log),
                "tool": "ERROR",
                "args": {},
                "result_summary": f"OpenAI call failed: {e}",
                "agent_reasoning": "Agent loop aborted — using data collected so far",
            })
            break

        msg = response.choices[0].message

        # Convert message to serialisable dict for the messages list
        msg_dict: Dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
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

            step_entry = {
                "step": len(trace_log),
                "tool": fn,
                "args": args,
                "result_summary": "",
                "agent_reasoning": msg.content or "",
            }

            if fn == "get_yfinance_data":
                fresh = _yfinance_fetch(args.get("ticker", ticker))
                if fresh.get("valid"):
                    accumulated["yfinance"] = fresh
                result_str = _yfinance_summary(fresh)
                step_entry["result_summary"] = f"yfinance refreshed | Price: ${fresh.get('price', 'N/A')}"
                if on_step:
                    on_step(f"Step {len(trace_log)}: refreshed Yahoo Finance data")

            elif fn == "get_sec_filing":
                if on_step:
                    on_step(f"Step {len(trace_log)}: fetching SEC EDGAR {args.get('filing_type', '10-K')}...")
                edgar = _edgar_fetch(args.get("ticker", ticker), args.get("filing_type", "10-K"))
                if edgar.get("available"):
                    accumulated["edgar"] = edgar
                    step_entry["result_summary"] = (
                        f"EDGAR {args.get('filing_type', '10-K')} verified | "
                        f"Revenue: ${(edgar.get('revenue') or 0) / 1e9:.1f}B | "
                        f"FCF: ${(edgar.get('free_cash_flow') or 0) / 1e9:.1f}B | "
                        f"EPS: {edgar.get('eps', 'N/A')}"
                    )
                else:
                    step_entry["result_summary"] = f"EDGAR unavailable — {edgar.get('error', 'unknown')}"
                result_str = _edgar_summary(edgar)

            elif fn == "web_search":
                query = args.get("query", "")
                if on_step:
                    on_step(f"Step {len(trace_log)}: searching — {query[:60]}...")
                text = _web_search(query)
                accumulated["web_searches"].append({"query": query, "result": text})
                step_entry["result_summary"] = f"Search: \"{query}\" → {text[:120]}..."
                result_str = text[:3000]

            else:
                result_str = "Unknown tool"
                step_entry["result_summary"] = "Unknown tool called"

            trace_log.append(step_entry)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str[:4000],
            })

    final_data = _merge_data(accumulated)
    return final_data, trace_log


def _merge_data(accumulated: Dict) -> Dict:
    """
    Merge yfinance (base) with EDGAR overrides.
    EDGAR numbers replace yfinance for matching financial statement fields
    because EDGAR data comes directly from official SEC filings.
    """
    base = accumulated.get("yfinance") or {}
    edgar = accumulated.get("edgar") or {}
    web = accumulated.get("web_searches") or []

    merged = dict(base)

    # Fields where EDGAR is authoritative — override yfinance values
    for edgar_key, merged_key in {
        "eps": "eps",
        "free_cash_flow": "free_cash_flow",
        "operating_cash_flow": "operating_cash_flow",
        "shares_outstanding": "shares_outstanding",
        "total_debt": "total_debt",
    }.items():
        if edgar.get(edgar_key) is not None:
            merged[merged_key] = edgar[edgar_key]

    # Attach raw sources for fact checker and research log UI
    merged["_edgar_data"] = edgar
    merged["_web_context"] = "\n\n".join(
        f"Query: {w['query']}\n{w['result']}" for w in web
    )
    merged["_data_sources"] = {
        "yfinance": base.get("valid", False),
        "edgar": edgar.get("available", False),
        "web_searches": len(web),
    }

    return merged


# ============== FACT CHECKER ==============

def run_fact_checker(context_str: str, raw_data: Dict) -> List[Dict]:
    """
    Compare key numbers mentioned in the analysis context string against raw_data.
    Returns a list of discrepancies where the difference exceeds 5%.
    """
    discrepancies = []

    # (label, raw_data_key, regex_pattern, scale_to_raw)
    # scale_to_raw: multiply the parsed number by this to get the raw unit
    checks = [
        ("P/E Ratio",      "pe_ratio",       r"P/E Ratio[:\s]+([\d.]+)",                1.0),
        ("Forward P/E",    "forward_pe",      r"Forward P/E[:\s]+([\d.]+)",              1.0),
        ("EPS",            "eps",             r"\bEPS[:\s]+\$([\d.]+)",                  1.0),
        ("Forward EPS",    "forward_eps",     r"Forward EPS[:\s]+\$([\d.]+)",            1.0),
        ("Beta",           "beta",            r"\bBeta[:\s]+([\d.]+)",                   1.0),
        ("Free Cash Flow", "free_cash_flow",  r"Free Cash Flow[:\s]+\$([\d.]+)B",        1e9),
        ("Market Cap",     "market_cap",      r"Market Cap[:\s]+\$([\d.]+)B",            1e9),
        ("ROE",            "roe",             r"\bROE[:\s]+([\d.]+)%",                   0.01),
    ]

    for label, key, pattern, scale in checks:
        raw_val = raw_data.get(key)
        if raw_val is None:
            continue
        match = re.search(pattern, context_str, re.IGNORECASE)
        if not match:
            continue
        try:
            reported_raw = float(match.group(1)) * scale
            actual = float(raw_val)
            if actual == 0:
                continue
            diff_pct = abs(reported_raw - actual) / abs(actual) * 100
            if diff_pct > 5:
                discrepancies.append({
                    "field": label,
                    "in_report": float(match.group(1)),
                    "in_raw_data": actual / scale,
                    "diff_pct": round(diff_pct, 1),
                    "unit": "B USD" if scale == 1e9 else ("%" if scale == 0.01 else ""),
                })
        except (ValueError, TypeError):
            continue

    return discrepancies


# ============== INSIDER TRADING (Form 4) ==============


def _empty_summary() -> Dict:
    """Return a zero-value insider summary dict for unavailable/empty cases."""
    return {
        "num_buys": 0,
        "num_sells": 0,
        "total_buy_value": 0.0,
        "total_sell_value": 0.0,
        "net_buy_value": 0.0,
        "unique_buyers": 0,
        "signal": "MIXED / NEUTRAL",
        "no_activity": True,
    }


def _parse_form4_xml(content: bytes, filing_date: str) -> List[Dict]:
    """
    Parse Form 4 XML bytes and return open-market buy/sell transactions.
    Excludes M (option exercise), F (tax withholding), A (award).
    Falls back to empty list on any parse error.
    """
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []

    insider_name = ""
    insider_title = ""
    for elem in root.iter():
        local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if local == "rptOwnerName" and elem.text and not insider_name:
            insider_name = elem.text.strip().title()
        if local == "officerTitle" and elem.text and not insider_title:
            insider_title = elem.text.strip()

    trades = []
    for txn in root.iter():
        local = txn.tag.split("}")[-1] if "}" in txn.tag else txn.tag
        if local != "nonDerivativeTransaction":
            continue

        code = ""
        date = filing_date
        shares = 0.0
        price = 0.0

        for child in txn.iter():
            ctag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if ctag == "transactionCode" and child.text:
                code = child.text.strip()
            elif ctag == "transactionDate":
                for sub in child.iter():
                    stag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                    if stag == "value" and sub.text:
                        date = sub.text.strip()
                        break
            elif ctag == "transactionShares":
                for sub in child.iter():
                    stag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                    if stag == "value" and sub.text:
                        try:
                            shares = abs(float(sub.text.strip()))
                        except ValueError:
                            pass
                        break
            elif ctag == "transactionPricePerShare":
                for sub in child.iter():
                    stag = sub.tag.split("}")[-1] if "}" in sub.tag else sub.tag
                    if stag == "value" and sub.text:
                        try:
                            price = float(sub.text.strip())
                        except ValueError:
                            pass
                        break

        if code not in ("P", "S") or shares <= 0:
            continue

        trades.append({
            "date": date,
            "insider": insider_name or "Unknown",
            "title": insider_title or "Director/Officer",
            "type": "BUY" if code == "P" else "SELL",
            "shares": int(shares),
            "price": price,
            "value": shares * price,
        })

    return trades


def _compute_insider_summary(trades: List[Dict]) -> Dict:
    """Compute aggregate stats and classify signal from a list of trade dicts."""
    if not trades:
        return _empty_summary()

    buys = [t for t in trades if t["type"] == "BUY"]
    sells = [t for t in trades if t["type"] == "SELL"]

    total_buy_value = sum(t["value"] for t in buys)
    total_sell_value = sum(t["value"] for t in sells)
    net_buy_value = total_buy_value - total_sell_value
    unique_buyers = len({t["insider"] for t in buys})

    if unique_buyers >= 3 and net_buy_value > 0:
        signal = "STRONG BUY SIGNAL"
    elif len(buys) >= 1 and net_buy_value > 0:
        signal = "BUY SIGNAL"
    elif len(sells) >= 3 and net_buy_value < 0:
        signal = "SELL SIGNAL"
    else:
        signal = "MIXED / NEUTRAL"

    return {
        "num_buys": len(buys),
        "num_sells": len(sells),
        "total_buy_value": total_buy_value,
        "total_sell_value": total_sell_value,
        "net_buy_value": net_buy_value,
        "unique_buyers": unique_buyers,
        "signal": signal,
        "no_activity": False,
    }


def fetch_insider_trades(cik: str, months: int = 6) -> Dict:
    """
    Fetch Form 4 insider filings from SEC EDGAR for the given CIK.
    Returns only open-market buys (P) and sells (S) within the time window.
    """
    cutoff = (datetime.utcnow() - timedelta(days=months * 30)).strftime("%Y-%m-%d")
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"

    try:
        resp = requests.get(url, headers=_EDGAR_HEADERS, timeout=10)
        if resp.status_code != 200:
            return {
                "available": False,
                "error": f"EDGAR submissions unavailable (HTTP {resp.status_code})",
                "months": months,
                "trades": [],
                "summary": _empty_summary(),
            }
        submissions = resp.json()
    except Exception as e:
        return {"available": False, "error": str(e), "months": months, "trades": [], "summary": _empty_summary()}

    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    form4_entries = [
        (dates[i], accessions[i], primary_docs[i])
        for i in range(len(forms))
        if i < len(dates) and i < len(accessions) and i < len(primary_docs)
        and forms[i] == "4" and dates[i] >= cutoff
    ][:15]

    trades: List[Dict] = []
    cik_int = int(cik)
    for _date, accession, primary_doc in form4_entries:
        acc_clean = accession.replace("-", "")
        xml_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_clean}/{primary_doc}"
        try:
            time.sleep(0.1)
            r = requests.get(xml_url, headers=_EDGAR_HEADERS, timeout=8)
            if r.status_code != 200:
                continue
            trades.extend(_parse_form4_xml(r.content, _date))
        except Exception:
            continue

    trades.sort(key=lambda t: t["date"], reverse=True)
    return {
        "available": True,
        "error": None,
        "cik": cik,
        "months": months,
        "trades": trades,
        "summary": _compute_insider_summary(trades),
    }
