import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))

from sa_peers import get_peer_tickers, fetch_peer_data, build_peer_table

_DUMMY_MAIN = {
    "valid": True, "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology",
    "price": 200.0, "pe_ratio": 29.0, "forward_pe": 25.0, "eps": 6.43,
    "revenue_growth": 0.08, "profit_margin": 0.25, "market_cap": 3e12,
    "debt_to_equity": 1.5,
    "info": {"sharesOutstanding": 15_500_000_000, "ebitda": 130_000_000_000,
             "totalDebt": 110_000_000_000, "totalCash": 165_000_000_000},
}

_DUMMY_PEER = {
    "valid": True, "ticker": "MSFT", "name": "Microsoft Corp.", "sector": "Technology",
    "price": 400.0, "pe_ratio": 35.0, "forward_pe": 30.0, "eps": 11.0,
    "revenue_growth": 0.15, "profit_margin": 0.35, "market_cap": 3e12,
    "debt_to_equity": 0.3,
    "info": {"sharesOutstanding": 7_500_000_000, "ebitda": 110_000_000_000,
             "totalDebt": 50_000_000_000, "totalCash": 80_000_000_000},
}


def test_get_peer_tickers_no_key_uses_fallback():
    # No API key → fallback lookup should return built-in suggestions, not empty
    result = get_peer_tickers("AAPL", "Technology", "Apple Inc.", api_key="")
    assert isinstance(result, list)
    assert len(result) > 0  # fallback always returns peers for known tickers
    assert "AAPL" not in result  # should never include the subject ticker


def test_fetch_peer_data_empty_list_returns_empty():
    result = fetch_peer_data([])
    assert result == []


def test_build_peer_table_main_ticker_first():
    df = build_peer_table("AAPL", _DUMMY_MAIN, [_DUMMY_PEER])
    assert df.iloc[0]["Ticker"] == "AAPL"
    assert df.iloc[1]["Ticker"] == "MSFT"


def test_build_peer_table_has_all_columns():
    df = build_peer_table("AAPL", _DUMMY_MAIN, [_DUMMY_PEER])
    required = {"Ticker", "Market Cap ($B)", "P/E (TTM)", "Forward P/E",
                "EV/EBITDA", "Rev Growth %", "Profit Margin %", "EPS", "Debt/Equity"}
    assert required.issubset(set(df.columns))


def test_build_peer_table_ev_ebitda_computed():
    df = build_peer_table("AAPL", _DUMMY_MAIN, [_DUMMY_PEER])
    # EV = 3e12 + 110e9 - 165e9 = 2.945e12; EBITDA=130e9 → EV/EBITDA ≈ 22.7
    aapl_ev = df[df["Ticker"] == "AAPL"]["EV/EBITDA"].iloc[0]
    assert aapl_ev is not None
    assert 20 < aapl_ev < 30


def test_build_peer_table_missing_ebitda_gives_none():
    peer_no_ebitda = dict(_DUMMY_PEER)
    peer_no_ebitda["info"] = {**_DUMMY_PEER["info"], "ebitda": None}
    df = build_peer_table("AAPL", _DUMMY_MAIN, [peer_no_ebitda])
    msft_ev = df[df["Ticker"] == "MSFT"]["EV/EBITDA"].iloc[0]
    assert msft_ev is None


def test_build_peer_table_no_peers_returns_one_row():
    df = build_peer_table("AAPL", _DUMMY_MAIN, [])
    assert len(df) == 1
    assert df.iloc[0]["Ticker"] == "AAPL"


def test_fetch_peer_data_filters_invalid(monkeypatch):
    import sa_research_agent

    def _fake_fetch(ticker):
        return {"valid": False, "ticker": ticker}

    monkeypatch.setattr(sa_research_agent, "_yfinance_fetch", _fake_fetch)
    result = fetch_peer_data(["FAKE"])
    assert result == []
