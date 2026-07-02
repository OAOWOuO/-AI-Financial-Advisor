import re
from typing import Dict, List, Optional

import pandas as pd


def get_peer_tickers(ticker: str, sector: str, company_name: str, api_key: str) -> List[str]:
    """
    Uses GPT-4o to suggest 4-5 well-known, analyst-covered peer companies.
    Returns list of ticker symbols. Returns [] on missing key or any error.
    """
    if not api_key:
        return []

    prompt = (
        f"List 4-5 publicly traded companies that equity analysts commonly compare to "
        f"{company_name} ({ticker}) in the {sector} sector.\n"
        "Requirements:\n"
        "- Listed on NYSE or NASDAQ\n"
        "- Market cap over $1B\n"
        "- Covered by at least 5 major analyst firms\n"
        "- Directly competitive with or in the same industry as the target company\n\n"
        "Respond with ONLY the ticker symbols, one per line, nothing else.\n"
        "Example output:\n"
        "MSFT\n"
        "GOOGL\n"
        "META"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=60,
            temperature=0,
        )
        raw = response.choices[0].message.content or ""
        # Extract valid-looking tickers: 1-5 uppercase letters, optionally with a dot (BRK.A)
        candidates = re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b", raw)
        seen = set()
        result = []
        for t in candidates:
            if t != ticker.upper() and t not in seen:
                seen.add(t)
                result.append(t)
        return result[:5]
    except Exception:
        return []


def fetch_peer_data(tickers: List[str]) -> List[Dict]:
    """
    Fetches yfinance data for each ticker. Returns only valid results.
    Imports _yfinance_fetch at call time to avoid circular imports.
    """
    if not tickers:
        return []

    from sa_research_agent import _yfinance_fetch

    results = []
    for t in tickers:
        try:
            data = _yfinance_fetch(t)
            if data.get("valid"):
                results.append(data)
        except Exception:
            pass
    return results


def build_peer_table(main_ticker: str, main_data: Dict, peer_data_list: List[Dict]) -> pd.DataFrame:
    """
    Builds a comparison DataFrame with 9 columns.
    Main ticker is always the first row. Raw numeric values (None for missing).
    """
    def _ev_ebitda(d: Dict) -> Optional[float]:
        info = d.get("info") or {}
        ebitda = info.get("ebitda")
        market_cap = d.get("market_cap")
        total_debt = info.get("totalDebt") or 0
        cash = info.get("totalCash") or 0
        if not ebitda or ebitda <= 0 or not market_cap or market_cap <= 0:
            return None
        ev = market_cap + total_debt - cash
        return round(ev / ebitda, 1) if ev > 0 else None

    def _row(d: Dict) -> Dict:
        market_cap = d.get("market_cap")
        rev_growth = d.get("revenue_growth")
        profit_margin = d.get("profit_margin")
        return {
            "Ticker": d.get("ticker", "N/A"),
            "Market Cap ($B)": round(market_cap / 1e9, 1) if market_cap else None,
            "P/E (TTM)": d.get("pe_ratio"),
            "Forward P/E": d.get("forward_pe"),
            "EV/EBITDA": _ev_ebitda(d),
            "Rev Growth %": round(rev_growth * 100, 1) if rev_growth is not None else None,
            "Profit Margin %": round(profit_margin * 100, 1) if profit_margin is not None else None,
            "EPS": d.get("eps"),
            "Debt/Equity": d.get("debt_to_equity"),
        }

    rows = [_row(main_data)] + [_row(d) for d in peer_data_list]
    df = pd.DataFrame(rows)
    # Convert to object dtype and replace NaN with None
    df = df.astype('object')
    df = df.where(pd.notna(df), None)
    return df
