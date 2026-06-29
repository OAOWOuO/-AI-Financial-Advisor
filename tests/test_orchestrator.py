import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))

from sa_orchestrator import run_fundamental_agent
from sa_orchestrator import run_catalyst_agent, run_macro_agent

_DUMMY_YF = {
    "valid": True, "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology",
    "price": 200.0, "pe_ratio": 29.0, "forward_pe": 25.0, "eps": 6.43,
    "free_cash_flow": 100e9, "revenue_growth": 0.08, "profit_margin": 0.25,
    "market_cap": 3e12, "debt_to_equity": 1.5, "current_ratio": 1.0,
}


def test_fundamental_agent_no_key_returns_error_dict():
    """Without an API key the agent should return a dict with error set."""
    result = run_fundamental_agent("AAPL", _DUMMY_YF, api_key="")
    assert "summary" in result
    assert "edgar_data" in result
    assert "yf_data" in result
    assert "trace" in result
    assert "error" in result
    assert isinstance(result["summary"], str)
    assert isinstance(result["trace"], list)


def test_catalyst_agent_no_key_returns_empty():
    result = run_catalyst_agent("AAPL", "Apple Inc.", api_key="")
    assert "summary" in result
    assert "web_searches" in result
    assert "trace" in result
    assert isinstance(result["web_searches"], list)


def test_macro_agent_no_key_returns_empty():
    result = run_macro_agent("AAPL", "Technology", api_key="")
    assert "summary" in result
    assert "web_searches" in result
    assert "trace" in result
    assert isinstance(result["trace"], list)
