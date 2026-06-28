import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))

import xml.etree.ElementTree as ET
from sa_research_agent import (
    _parse_form4_xml,
    _compute_insider_summary,
    _empty_summary,
)
from sa_research_agent import _deterministic_insider_summary, analyze_insider_signal

# Minimal Form 4 XML with one BUY (code P) and one SELL (code S)
_FIXTURE_XML = b"""<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>John Smith</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship>
      <isOfficer>1</isOfficer>
      <officerTitle>CEO</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTransaction>
    <transactionDate><value>2026-05-01</value></transactionDate>
    <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
    <transactionAmounts>
      <transactionShares><value>1000</value></transactionShares>
      <transactionPricePerShare><value>150.00</value></transactionPricePerShare>
    </transactionAmounts>
  </nonDerivativeTransaction>
  <nonDerivativeTransaction>
    <transactionDate><value>2026-04-01</value></transactionDate>
    <transactionCoding><transactionCode>S</transactionCode></transactionCoding>
    <transactionAmounts>
      <transactionShares><value>500</value></transactionShares>
      <transactionPricePerShare><value>140.00</value></transactionPricePerShare>
    </transactionAmounts>
  </nonDerivativeTransaction>
  <nonDerivativeTransaction>
    <transactionDate><value>2026-03-01</value></transactionDate>
    <transactionCoding><transactionCode>M</transactionCode></transactionCoding>
    <transactionAmounts>
      <transactionShares><value>2000</value></transactionShares>
      <transactionPricePerShare><value>10.00</value></transactionPricePerShare>
    </transactionAmounts>
  </nonDerivativeTransaction>
</ownershipDocument>"""

_FIXTURE_XML_BAD = b"<not valid xml"


def test_parse_form4_xml_extracts_buy_and_sell():
    trades = _parse_form4_xml(_FIXTURE_XML, "2026-05-01")
    assert len(trades) == 2
    types = {t["type"] for t in trades}
    assert types == {"BUY", "SELL"}


def test_parse_form4_xml_excludes_option_exercise():
    trades = _parse_form4_xml(_FIXTURE_XML, "2026-05-01")
    assert all(t["type"] in ("BUY", "SELL") for t in trades)


def test_parse_form4_xml_buy_values():
    trades = _parse_form4_xml(_FIXTURE_XML, "2026-05-01")
    buy = next(t for t in trades if t["type"] == "BUY")
    assert buy["shares"] == 1000
    assert abs(buy["price"] - 150.0) < 0.01
    assert abs(buy["value"] - 150000.0) < 0.01
    assert buy["insider"] == "John Smith"
    assert buy["title"] == "CEO"


def test_parse_form4_xml_bad_xml_returns_empty():
    trades = _parse_form4_xml(_FIXTURE_XML_BAD, "2026-05-01")
    assert trades == []


def test_compute_insider_summary_strong_buy():
    trades = [
        {"type": "BUY", "insider": "Alice", "value": 500_000},
        {"type": "BUY", "insider": "Bob",   "value": 400_000},
        {"type": "BUY", "insider": "Carol", "value": 300_000},
    ]
    summary = _compute_insider_summary(trades)
    assert summary["signal"] == "STRONG BUY SIGNAL"
    assert summary["unique_buyers"] == 3
    assert summary["num_buys"] == 3
    assert summary["num_sells"] == 0
    assert not summary["no_activity"]


def test_compute_insider_summary_sell_signal():
    trades = [
        {"type": "SELL", "insider": "Alice", "value": 1_000_000},
        {"type": "SELL", "insider": "Bob",   "value": 800_000},
        {"type": "SELL", "insider": "Carol", "value": 600_000},
    ]
    summary = _compute_insider_summary(trades)
    assert summary["signal"] == "SELL SIGNAL"
    assert summary["num_sells"] == 3
    assert summary["net_buy_value"] < 0


def test_compute_insider_summary_no_activity():
    summary = _compute_insider_summary([])
    assert summary["no_activity"] is True
    assert summary["signal"] == "MIXED / NEUTRAL"


def test_empty_summary_shape():
    s = _empty_summary()
    required = {"num_buys", "num_sells", "total_buy_value", "total_sell_value",
                "net_buy_value", "unique_buyers", "signal", "no_activity"}
    assert required.issubset(s.keys())


def test_deterministic_no_activity():
    s = _empty_summary()
    text = _deterministic_insider_summary(s, 6)
    assert "No open-market insider trading" in text
    assert "6 months" in text


def test_deterministic_strong_buy():
    s = {
        "num_buys": 4, "num_sells": 0, "total_buy_value": 2_100_000,
        "total_sell_value": 0, "net_buy_value": 2_100_000,
        "unique_buyers": 3, "signal": "STRONG BUY SIGNAL", "no_activity": False,
    }
    text = _deterministic_insider_summary(s, 6)
    assert "3" in text  # unique buyers
    assert "2.1" in text  # total_buy_value in millions


def test_deterministic_sell_signal():
    s = {
        "num_buys": 0, "num_sells": 4, "total_buy_value": 0,
        "total_sell_value": 3_000_000, "net_buy_value": -3_000_000,
        "unique_buyers": 0, "signal": "SELL SIGNAL", "no_activity": False,
    }
    text = _deterministic_insider_summary(s, 6)
    assert "sold" in text.lower() or "sell" in text.lower() or "sale" in text.lower()


def test_analyze_insider_signal_no_key_uses_deterministic():
    trades_data = {
        "available": True,
        "months": 6,
        "trades": [],
        "summary": _empty_summary(),
    }
    result = analyze_insider_signal(trades_data, "Apple Inc", api_key="")
    assert isinstance(result, str)
    assert len(result) > 20


def _insider_raw_score_reference(summary: dict) -> int:
    """Reference implementation matching what stock_analyzer.py will contain."""
    if not summary or summary.get("no_activity"):
        return 6
    unique_buyers = summary.get("unique_buyers", 0)
    num_buys = summary.get("num_buys", 0)
    num_sells = summary.get("num_sells", 0)
    net_buy_value = summary.get("net_buy_value", 0.0)
    net_sell_value = -net_buy_value if net_buy_value < 0 else 0.0
    total_buy_value = summary.get("total_buy_value", 0.0)

    if unique_buyers >= 3 and total_buy_value > 1_000_000:
        return 20
    elif num_buys >= 1 and net_buy_value > 0:
        return 12
    elif num_sells >= 3 and net_sell_value > 2_000_000:
        return -5
    elif net_sell_value > 0:
        return 2
    else:
        return 6


def test_insider_raw_score_strong_buy():
    s = {"unique_buyers": 3, "num_buys": 4, "num_sells": 0,
         "total_buy_value": 2_000_001, "net_buy_value": 2_000_001,
         "no_activity": False}
    assert _insider_raw_score_reference(s) == 20


def test_insider_raw_score_buy():
    s = {"unique_buyers": 1, "num_buys": 1, "num_sells": 0,
         "total_buy_value": 50_000, "net_buy_value": 50_000,
         "no_activity": False}
    assert _insider_raw_score_reference(s) == 12


def test_insider_raw_score_no_activity():
    assert _insider_raw_score_reference({}) == 6
    assert _insider_raw_score_reference({"no_activity": True}) == 6


def test_insider_raw_score_heavy_sell():
    s = {"unique_buyers": 0, "num_buys": 0, "num_sells": 4,
         "total_buy_value": 0, "net_buy_value": -3_000_000,
         "no_activity": False}
    assert _insider_raw_score_reference(s) == -5


def test_insider_scaled_score_at_15pct():
    raw = 20
    weight_pct = 15
    scaled = raw * (weight_pct / 20.0)
    assert abs(scaled - 15.0) < 0.001


def test_total_max_sums_to_100():
    weight_pct = 15
    scale = (100 - weight_pct) / 100
    val_max = 30 * scale
    prof_max = 25 * scale
    growth_max = 25 * scale
    health_max = 20 * scale
    insider_max = float(weight_pct)
    total = val_max + prof_max + growth_max + health_max + insider_max
    assert abs(total - 100.0) < 0.001
