import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "streamlit_app"))

import xml.etree.ElementTree as ET
from sa_research_agent import (
    _parse_form4_xml,
    _compute_insider_summary,
    _empty_summary,
)

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
