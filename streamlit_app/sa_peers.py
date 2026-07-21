import re
from typing import Dict, List, Optional

import pandas as pd


# Hardcoded peer suggestions — used as fallback when no API key is available
_TICKER_PEERS: Dict[str, List[str]] = {
    # Mega-cap tech
    "AAPL": ["MSFT", "GOOGL", "META", "AMZN", "NVDA"],
    "MSFT": ["AAPL", "GOOGL", "AMZN", "META", "ORCL"],
    "GOOGL": ["META", "MSFT", "AAPL", "AMZN", "SNAP"],
    "GOOG": ["META", "MSFT", "AAPL", "AMZN", "SNAP"],
    "META": ["GOOGL", "SNAP", "PINS", "MSFT", "NFLX"],
    "AMZN": ["MSFT", "GOOGL", "WMT", "SHOP", "EBAY"],
    # Semiconductors
    "NVDA": ["AMD", "INTC", "QCOM", "AVGO", "MRVL"],
    "AMD": ["NVDA", "INTC", "QCOM", "AVGO", "MRVL"],
    "INTC": ["AMD", "NVDA", "QCOM", "AVGO", "TXN"],
    "QCOM": ["AVGO", "MRVL", "INTC", "AMD", "TXN"],
    "AVGO": ["QCOM", "MRVL", "INTC", "AMD", "TXN"],
    "MRVL": ["AVGO", "QCOM", "NVDA", "AMD", "INTC"],
    "TSM": ["INTC", "SSNLF", "UMC", "ASML", "AMAT"],
    "ASML": ["AMAT", "LRCX", "KLAC", "TSM", "INTC"],
    # Software / SaaS
    "CRM": ["NOW", "ADBE", "ORCL", "SAP", "WDAY"],
    "ADBE": ["CRM", "NOW", "ORCL", "MSFT", "WDAY"],
    "NOW": ["CRM", "ADBE", "WDAY", "ORCL", "MSFT"],
    "WDAY": ["NOW", "CRM", "ADBE", "ORCL", "SAP"],
    "ORCL": ["MSFT", "CRM", "SAP", "NOW", "WDAY"],
    "SNOW": ["DBRX", "MDB", "PLTR", "MSFT", "GOOGL"],
    "PLTR": ["SNOW", "MDB", "CRM", "MSFT", "NOW"],
    "MDB": ["SNOW", "DBRX", "ORCL", "MSFT", "PLTR"],
    # Internet / streaming
    "NFLX": ["DIS", "WBD", "PARA", "AMZN", "AAPL"],
    "DIS": ["NFLX", "WBD", "PARA", "CMCSA", "FOXA"],
    "SPOT": ["AAPL", "GOOGL", "AMZN", "TMUS", "SIRIUS"],
    # E-commerce / retail
    "SHOP": ["AMZN", "EBAY", "BIGC", "WIX", "ETSY"],
    "EBAY": ["AMZN", "SHOP", "ETSY", "WMT", "TGT"],
    "ETSY": ["EBAY", "AMZN", "SHOP", "TGT", "WMT"],
    "WMT": ["TGT", "COST", "AMZN", "KR", "DG"],
    "TGT": ["WMT", "COST", "AMZN", "KR", "DG"],
    "COST": ["WMT", "TGT", "BJ", "AMZN", "KR"],
    # EV / auto
    "TSLA": ["GM", "F", "RIVN", "NIO", "STLA"],
    "RIVN": ["TSLA", "LCID", "NIO", "GM", "F"],
    "NIO": ["TSLA", "XPEV", "LI", "RIVN", "GM"],
    "GM": ["F", "STLA", "TSLA", "TM", "HMC"],
    "F": ["GM", "STLA", "TSLA", "TM", "HMC"],
    # Finance
    "JPM": ["BAC", "WFC", "GS", "MS", "C"],
    "BAC": ["JPM", "WFC", "GS", "MS", "C"],
    "WFC": ["JPM", "BAC", "GS", "MS", "USB"],
    "GS": ["MS", "JPM", "BAC", "WFC", "BX"],
    "MS": ["GS", "JPM", "BAC", "WFC", "BX"],
    "BX": ["KKR", "CG", "APO", "GS", "MS"],
    "V": ["MA", "AXP", "PYPL", "FIS", "FI"],
    "MA": ["V", "AXP", "PYPL", "FIS", "FI"],
    "PYPL": ["V", "MA", "SQ", "AFRM", "SOFI"],
    "SQ": ["PYPL", "MA", "V", "AFRM", "SOFI"],
    # Healthcare / pharma
    "JNJ": ["PFE", "MRK", "ABBV", "UNH", "ABT"],
    "PFE": ["JNJ", "MRK", "ABBV", "BMY", "LLY"],
    "MRK": ["PFE", "JNJ", "ABBV", "BMY", "LLY"],
    "ABBV": ["MRK", "PFE", "JNJ", "BMY", "LLY"],
    "LLY": ["NVO", "ABBV", "PFE", "MRK", "BMY"],
    "NVO": ["LLY", "ABBV", "PFE", "MRK", "AZN"],
    "UNH": ["CVS", "CI", "HUM", "CNC", "MOH"],
    "CVS": ["WBA", "UNH", "MCK", "CI", "HUM"],
    # Energy
    "XOM": ["CVX", "COP", "EOG", "SLB", "BP"],
    "CVX": ["XOM", "COP", "EOG", "PXD", "SLB"],
    "COP": ["XOM", "CVX", "EOG", "PXD", "DVN"],
    # Consumer
    "MCD": ["QSR", "YUM", "CMG", "DPZ", "WEN"],
    "SBUX": ["MCD", "CMG", "QSR", "DPZ", "DNUT"],
    "CMG": ["MCD", "SBUX", "QSR", "YUM", "DPZ"],
    "KO": ["PEP", "MNST", "CELH", "CCEP", "KDP"],
    "PEP": ["KO", "MNST", "CELH", "CCEP", "KDP"],
    "NKE": ["ADDYY", "UAA", "SKX", "PVH", "VFC"],
    "PG": ["CL", "KMB", "CHD", "CLX", "EL"],
    # Ride-share / travel
    "UBER": ["LYFT", "DASH", "GRAB", "BKNG", "EXPE"],
    "LYFT": ["UBER", "DASH", "GRAB", "BKNG", "EXPE"],
    "BKNG": ["EXPE", "ABNB", "TRIP", "UBER", "HLT"],
    "ABNB": ["BKNG", "EXPE", "TRIP", "MAR", "HLT"],
    # Airlines
    "DAL": ["UAL", "AAL", "LUV", "ALK", "JBLU"],
    "UAL": ["DAL", "AAL", "LUV", "ALK", "JBLU"],
    "AAL": ["DAL", "UAL", "LUV", "ALK", "JBLU"],
    # Telecom
    "T": ["VZ", "TMUS", "CMCSA", "CHTR", "DISH"],
    "VZ": ["T", "TMUS", "CMCSA", "CHTR", "DISH"],
    "TMUS": ["T", "VZ", "CMCSA", "CHTR", "DISH"],
    # Real estate / REITs
    "AMT": ["CCI", "SBAC", "EQIX", "DLR", "PLD"],
    "EQIX": ["DLR", "AMT", "CCI", "CONE", "QTS"],
    # Crypto / fintech
    "COIN": ["MSTR", "HOOD", "SQ", "PYPL", "MARA"],
}

_SECTOR_PEERS: Dict[str, List[str]] = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "Communication Services": ["META", "GOOGL", "NFLX", "DIS", "CMCSA"],
    "Consumer Cyclical": ["AMZN", "TSLA", "MCD", "NKE", "BKNG"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS"],
    "Financial Services": ["JPM", "BAC", "V", "MA", "GS"],
    "Healthcare": ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
    "Industrials": ["CAT", "HON", "GE", "RTX", "UPS"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Basic Materials": ["LIN", "APD", "ECL", "DD", "NEM"],
    "Real Estate": ["AMT", "PLD", "EQIX", "SPG", "CCI"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
}


def _fallback_peers(ticker: str, sector: str) -> List[str]:
    """Return hardcoded peer suggestions without any API call."""
    ticker_upper = ticker.upper()
    if ticker_upper in _TICKER_PEERS:
        return [t for t in _TICKER_PEERS[ticker_upper] if t != ticker_upper][:5]
    sector_key = (sector or "").strip()
    for key, peers in _SECTOR_PEERS.items():
        if key.lower() in sector_key.lower() or sector_key.lower() in key.lower():
            return [t for t in peers if t != ticker_upper][:5]
    return []


def get_peer_tickers(ticker: str, sector: str, company_name: str, api_key: str) -> List[str]:
    """
    Returns 4-5 peer tickers. Uses GPT-4o when key is available,
    falls back to a hardcoded lookup otherwise.
    """
    if api_key:
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
            candidates = re.findall(r"\b[A-Z]{1,5}(?:\.[A-Z])?\b", raw)
            seen = set()
            result = []
            for t in candidates:
                if t != ticker.upper() and t not in seen:
                    seen.add(t)
                    result.append(t)
            if result:
                return result[:5]
        except Exception:
            pass

    return _fallback_peers(ticker, sector)


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
    df = df.astype("object")
    df = df.where(pd.notna(df), None)
    return df
