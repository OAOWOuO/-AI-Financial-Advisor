import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))

from sa_orchestrator import run_fundamental_agent
from sa_orchestrator import run_catalyst_agent, run_macro_agent
from sa_orchestrator import run_orchestrator
from sa_orchestrator import run_multi_agent_research
from sa_orchestrator import run_valuation_agent

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


def test_orchestrator_no_key_concatenates_summaries():
    fundamental = {"summary": "Strong FCF of $100B.", "edgar_data": {}, "yf_data": {}, "trace": [], "error": None}
    catalyst    = {"summary": "Earnings beat by 5%.", "web_searches": [], "trace": [], "error": None}
    macro       = {"summary": "Tech sector favourable.", "web_searches": [], "trace": [], "error": None}
    raw = {"price": 200.0, "market_cap": 3e12}
    result = run_orchestrator("AAPL", fundamental, catalyst, macro, raw, api_key="")
    assert "Strong FCF" in result
    assert "Earnings beat" in result
    assert "Tech sector" in result


def test_orchestrator_empty_reports_returns_empty_string():
    fundamental = {"summary": "", "edgar_data": {}, "yf_data": {}, "trace": [], "error": "No API key"}
    catalyst    = {"summary": "", "web_searches": [], "trace": [], "error": "No API key"}
    macro       = {"summary": "", "web_searches": [], "trace": [], "error": "No API key"}
    raw = {"price": 100.0, "market_cap": 1e12}
    result = run_orchestrator("XYZ", fundamental, catalyst, macro, raw, api_key="")
    assert isinstance(result, str)


_DUMMY_YF_VALID = {
    "valid": True, "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology",
    "price": 200.0, "pe_ratio": 29.0, "forward_pe": 25.0, "eps": 6.43,
    "free_cash_flow": 100e9, "revenue_growth": 0.08, "profit_margin": 0.25,
    "market_cap": 3e12, "debt_to_equity": 1.5, "current_ratio": 1.0,
    "hist_1y": None, "hist_2y": None, "hist_5y": None,
    "info": {}, "income_stmt": None, "balance_sheet": None, "cash_flow": None,
    "quarterly_income": None, "quarterly_bs": None,
}


def test_multi_agent_no_key_returns_pipeline_result():
    """No API key → routes to _run_pipeline, returns valid data dict and trace."""
    data, trace = run_multi_agent_research("AAPL", api_key="")
    assert isinstance(data, dict)
    assert isinstance(trace, list)
    assert len(trace) >= 1


def test_multi_agent_result_has_orchestrator_thesis_key():
    """Even without API key, the data dict must have _orchestrator_thesis key."""
    data, trace = run_multi_agent_research("AAPL", api_key="")
    assert "_orchestrator_thesis" in data


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
