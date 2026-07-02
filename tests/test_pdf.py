import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))

from sa_pdf import generate_pdf

_DUMMY_DATA = {
    "valid": True, "ticker": "AAPL", "name": "Apple Inc.", "sector": "Technology",
    "price": 200.0, "pe_ratio": 29.0, "forward_pe": 25.0, "eps": 6.43,
    "revenue_growth": 0.08, "profit_margin": 0.25, "market_cap": 3e12,
    "debt_to_equity": 1.5,
    "info": {"sharesOutstanding": 15_500_000_000, "ebitda": 130_000_000_000,
             "totalDebt": 110_000_000_000, "totalCash": 165_000_000_000},
    "_orchestrator_thesis": "Apple Inc. demonstrates strong financial health with robust cash generation.",
    "_valuation": {
        "intrinsic_value": 180.0,
        "ev_ebitda": 22.7,
        "summary": "Based on DCF analysis, AAPL appears modestly overvalued at current prices.",
    },
    "_peer_data": [],
    "_insider_signal_text": "Recent insider activity shows modest buying from C-suite executives.",
}

_DUMMY_VALUATION = {
    "pe_valuation": {"low": 150.0, "mid": 180.0, "high": 210.0},
    "forward_pe_valuation": {"low": 155.0, "mid": 185.0, "high": 215.0},
}


def test_generate_pdf_returns_bytes():
    result = generate_pdf(_DUMMY_DATA, _DUMMY_VALUATION)
    assert isinstance(result, bytes)
    assert len(result) > 1000


def test_generate_pdf_starts_with_pdf_header():
    result = generate_pdf(_DUMMY_DATA, _DUMMY_VALUATION)
    assert result[:4] == b"%PDF"


def test_generate_pdf_no_peer_data():
    data = dict(_DUMMY_DATA)
    data["_peer_data"] = []
    result = generate_pdf(data, _DUMMY_VALUATION)
    assert result[:4] == b"%PDF"


def test_generate_pdf_no_valuation_data():
    data = dict(_DUMMY_DATA)
    data["_valuation"] = {}
    result = generate_pdf(data, {})
    assert result[:4] == b"%PDF"


def test_generate_pdf_no_orchestrator_thesis():
    data = dict(_DUMMY_DATA)
    data.pop("_orchestrator_thesis", None)
    result = generate_pdf(data, _DUMMY_VALUATION)
    assert result[:4] == b"%PDF"
