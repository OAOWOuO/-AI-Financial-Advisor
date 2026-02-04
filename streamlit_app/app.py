"""
AI Hedge Fund Terminal v4.8
Fixes: Holdings impact, analyst signals display, news section, settings reorganization
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="AI Hedge Fund Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============== CSS - COMPREHENSIVE DARK THEME ==============
st.markdown("""
<style>
    /* Base dark theme */
    .main { background: #0d1117; }
    .stApp { background: #0d1117; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Text styling */
    h1, h2, h3, h4 { color: #e6edf3 !important; font-weight: 600 !important; }
    p, span, label, li, div { color: #c9d1d9 !important; }

    /* Remove code-style boxes */
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        font-family: inherit !important;
    }
    code {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        color: inherit !important;
        font-family: inherit !important;
        font-size: inherit !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #161b22; padding: 8px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #8b949e; border-radius: 6px; padding: 8px 16px; }
    .stTabs [aria-selected="true"] { background: #238636 !important; color: white !important; }

    /* Expander styling - simplified to prevent flickering */
    [data-testid="stExpander"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        transition: none !important;
    }
    [data-testid="stExpander"] > details {
        background: #161b22 !important;
        transition: none !important;
    }
    [data-testid="stExpander"] > details > summary {
        background: #161b22 !important;
        color: #e6edf3 !important;
        transition: none !important;
    }
    [data-testid="stExpander"] > details > summary:hover {
        background: #21262d !important;
    }
    [data-testid="stExpander"] > details > div {
        background: #0d1117 !important;
        border-top: 1px solid #30363d !important;
        transition: none !important;
    }
    /* Prevent layout shift on expand/collapse */
    [data-testid="stExpander"] * {
        animation: none !important;
    }

    /* Button styling */
    .stButton > button {
        background: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background: #30363d !important;
        border-color: #8b949e !important;
    }
    .stButton > button[kind="primary"] {
        background: #238636 !important;
        color: white !important;
        border: 1px solid #238636 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #2ea043 !important;
    }
    .stButton > button[kind="secondary"] {
        background: #21262d !important;
        color: #c9d1d9 !important;
    }

    /* Selectbox / dropdown styling */
    [data-baseweb="select"] {
        background: #161b22 !important;
    }
    [data-baseweb="select"] > div {
        background: #161b22 !important;
        border-color: #30363d !important;
        color: #c9d1d9 !important;
    }
    [data-baseweb="popover"] {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    [data-baseweb="menu"] {
        background: #161b22 !important;
    }
    [data-baseweb="menu"] li {
        background: #161b22 !important;
        color: #c9d1d9 !important;
    }
    [data-baseweb="menu"] li:hover {
        background: #21262d !important;
    }
    [role="listbox"] {
        background: #161b22 !important;
    }
    [role="option"] {
        background: #161b22 !important;
        color: #c9d1d9 !important;
    }
    [role="option"]:hover {
        background: #21262d !important;
    }

    /* Input styling */
    .stTextInput > div > div > input {
        background: #161b22 !important;
        color: #c9d1d9 !important;
        border-color: #30363d !important;
    }
    .stTextArea > div > div > textarea {
        background: #161b22 !important;
        color: #c9d1d9 !important;
        border-color: #30363d !important;
    }
    .stNumberInput > div > div > input {
        background: #161b22 !important;
        color: #c9d1d9 !important;
        border-color: #30363d !important;
    }

    /* Checkbox styling */
    .stCheckbox > label {
        color: #c9d1d9 !important;
    }
    .stCheckbox > label > span {
        color: #c9d1d9 !important;
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background: #30363d !important;
    }

    /* Alert boxes */
    .stAlert {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
    }
    [data-baseweb="notification"] {
        background: #161b22 !important;
    }

    /* Info/Warning/Success/Error boxes */
    .stInfo, [data-testid="stNotification"][data-type="info"] {
        background: #0d2237 !important;
        border: 1px solid #58a6ff !important;
    }
    .stWarning, [data-testid="stNotification"][data-type="warning"] {
        background: #3d2a1f !important;
        border: 1px solid #d29922 !important;
    }
    .stSuccess, [data-testid="stNotification"][data-type="success"] {
        background: #1f3d2a !important;
        border: 1px solid #3fb950 !important;
    }
    .stError, [data-testid="stNotification"][data-type="error"] {
        background: #3d1f1f !important;
        border: 1px solid #f85149 !important;
    }

    /* Dataframe styling */
    [data-testid="stDataFrame"] { pointer-events: none; }
    .stDataFrame {
        background: #161b22 !important;
    }

    /* Download button */
    .stDownloadButton > button {
        background: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
    }
    .stDownloadButton > button:hover {
        background: #30363d !important;
    }

    /* Caption styling */
    .stCaption {
        color: #8b949e !important;
    }

    /* Divider */
    hr {
        border-color: #30363d !important;
    }
</style>
""", unsafe_allow_html=True)


# ============== ALLOCATION MODES ==============
ALLOCATION_MODES = {
    "max_deploy": {"name": "Maximum Deployment", "desc": "Deploy 95%+ of capital. Position caps scale with # of tickers.", "target_pct": 0.95},
    "equal_weight": {"name": "Equal Weight", "desc": "Split capital equally among actionable tickers.", "target_pct": 0.90},
    "confidence_weighted": {"name": "Confidence Weighted", "desc": "Size by conviction. May hold cash.", "target_pct": None},
    "conservative": {"name": "Conservative", "desc": "High-conviction only. 30-50% cash buffer.", "target_pct": 0.50}
}

# ============== ANALYST DEFINITIONS ==============
ANALYST_CATEGORIES = {
    "Value Investors": {
        "warren_buffett": {"name": "Warren Buffett", "desc": "Moats, quality management, long-term value", "bias": -0.1,
            "thesis": "Seeks companies with durable competitive advantages (moats) trading below intrinsic value",
            "drivers": "Strong brand, pricing power, consistent earnings, quality management",
            "horizon": "5-10+ years", "risks": "May miss growth opportunities, slow to act"},
        "charlie_munger": {"name": "Charlie Munger", "desc": "Mental models, business quality", "bias": -0.1,
            "thesis": "Focus on business quality and management integrity over pure value metrics",
            "drivers": "Mental models, avoiding stupidity, quality over price",
            "horizon": "Long-term", "risks": "Concentration risk, patience required"},
        "ben_graham": {"name": "Benjamin Graham", "desc": "Margin of safety, net-net value", "bias": -0.15,
            "thesis": "Deep value requiring significant discount to book value",
            "drivers": "Net current asset value, margin of safety, quantitative screens",
            "horizon": "1-3 years", "risks": "Value traps, declining businesses"},
        "joel_greenblatt": {"name": "Joel Greenblatt", "desc": "Magic formula: ROIC + earnings yield", "bias": -0.05,
            "thesis": "Quantitative value screening based on return on capital and earnings yield",
            "drivers": "High ROIC, high earnings yield, systematic approach",
            "horizon": "1-2 years", "risks": "Sector concentration, mechanical approach"},
        "seth_klarman": {"name": "Seth Klarman", "desc": "Deep value, distressed assets", "bias": -0.15,
            "thesis": "Contrarian deep value in distressed or out-of-favor situations",
            "drivers": "Extreme pessimism, catalyst identification, risk management",
            "horizon": "2-5 years", "risks": "Timing uncertainty, permanent capital loss"},
    },
    "Growth Investors": {
        "peter_lynch": {"name": "Peter Lynch", "desc": "PEG ratio, growth at reasonable price", "bias": 0.05,
            "thesis": "Growth at a reasonable price, invest in what you understand",
            "drivers": "PEG ratio, local knowledge, growth sustainability",
            "horizon": "3-5 years", "risks": "Overpaying for growth, diversification"},
        "phil_fisher": {"name": "Philip Fisher", "desc": "Scuttlebutt, quality growth", "bias": 0.05,
            "thesis": "Long-term growth investing with deep qualitative research",
            "drivers": "Management quality, R&D, competitive position, scuttlebutt",
            "horizon": "10+ years", "risks": "Concentration, qualitative biases"},
        "cathie_wood": {"name": "Cathie Wood", "desc": "Disruptive innovation, exponential growth", "bias": 0.2,
            "thesis": "High-conviction bets on disruptive innovation and exponential growth",
            "drivers": "Disruption potential, TAM expansion, Wright's Law cost curves",
            "horizon": "5+ years", "risks": "Volatility, valuation, execution risk"},
        "bill_ackman": {"name": "Bill Ackman", "desc": "Activist catalysts, concentrated bets", "bias": 0.1,
            "thesis": "Concentrated positions with activist catalysts to unlock value",
            "drivers": "Undervalued assets, activist engagement, management change",
            "horizon": "2-4 years", "risks": "Concentration, activism resistance"},
    },
    "Macro Traders": {
        "stanley_druckenmiller": {"name": "Stanley Druckenmiller", "desc": "Macro trends, asymmetric bets", "bias": 0,
            "thesis": "Macro trend following with aggressive sizing on high-conviction ideas",
            "drivers": "Liquidity cycles, central bank policy, asymmetric setups",
            "horizon": "Months to years", "risks": "Timing, leverage"},
        "george_soros": {"name": "George Soros", "desc": "Reflexivity, regime changes", "bias": 0,
            "thesis": "Identifies reflexive feedback loops and regime changes",
            "drivers": "Market psychology, reflexivity, boom-bust cycles",
            "horizon": "Variable", "risks": "Timing, complexity"},
        "ray_dalio": {"name": "Ray Dalio", "desc": "Economic machine, risk parity", "bias": -0.05,
            "thesis": "Systematic macro based on economic machine principles",
            "drivers": "Debt cycles, productivity, diversification",
            "horizon": "Full cycle", "risks": "Model assumptions, correlation breakdown"},
        "paul_tudor_jones": {"name": "Paul Tudor Jones", "desc": "Technical macro, trend following", "bias": 0,
            "thesis": "Technical analysis combined with macro themes",
            "drivers": "Price action, trend, sentiment extremes",
            "horizon": "Weeks to months", "risks": "Whipsaws, false signals"},
    },
    "Quantitative Agents": {
        "fundamentals_agent": {"name": "Fundamentals Analyst", "desc": "Financial ratios, earnings quality", "bias": 0,
            "thesis": "Analyzes financial statements, ratios, and earnings quality",
            "drivers": "Revenue growth, margins, ROE, debt levels, cash flow",
            "horizon": "1-2 years", "risks": "Backward looking, accounting manipulation"},
        "technical_agent": {"name": "Technical Analyst", "desc": "Price patterns, momentum, RSI, MACD", "bias": 0,
            "thesis": "Technical indicators and price pattern analysis",
            "drivers": "RSI, MACD, moving averages, support/resistance",
            "horizon": "Days to weeks", "risks": "False signals, changing regimes"},
        "sentiment_agent": {"name": "Sentiment Analyst", "desc": "News sentiment, social media", "bias": 0.05,
            "thesis": "Aggregates news sentiment, social media buzz, analyst ratings",
            "drivers": "News flow, social sentiment, analyst revisions",
            "horizon": "Days to months", "risks": "Noise, manipulation, lag"},
        "valuation_agent": {"name": "Valuation Analyst", "desc": "DCF, comparable analysis", "bias": -0.05,
            "thesis": "DCF models, comparable company analysis, sum-of-parts",
            "drivers": "Intrinsic value, multiples, growth assumptions",
            "horizon": "1-3 years", "risks": "Model sensitivity, assumptions"},
        "momentum_agent": {"name": "Momentum Analyst", "desc": "Price and earnings momentum", "bias": 0.1,
            "thesis": "Follows price and earnings momentum trends",
            "drivers": "Price momentum, earnings surprises, estimate revisions",
            "horizon": "3-12 months", "risks": "Reversals, crowding"},
        "risk_agent": {"name": "Risk Analyst", "desc": "Volatility, drawdown, tail risk", "bias": -0.1,
            "thesis": "Focuses on risk metrics and downside protection",
            "drivers": "Volatility, drawdown, VaR, correlation, tail risk",
            "horizon": "Ongoing", "risks": "Over-conservatism, missed upside"},
    },
}


def get_all_analysts():
    result = {}
    for cat, analysts in ANALYST_CATEGORIES.items():
        for key, info in analysts.items():
            result[key] = {**info, "category": cat}
    return result

ALL_ANALYSTS = get_all_analysts()
ALL_ANALYST_KEYS = list(ALL_ANALYSTS.keys())


# ============== RISK PARAMETERS ==============
def get_risk_params(risk_level: float, custom: dict = None) -> dict:
    params = {
        "max_position_pct": {"value": round(10 + risk_level * 25, 1), "unit": "%", "desc": "Max allocation per position"},
        "stop_loss_pct": {"value": round(20 - risk_level * 15, 1), "unit": "%", "desc": "Stop loss distance"},
        "take_profit_pct": {"value": round(15 + risk_level * 45, 1), "unit": "%", "desc": "Take profit target"},
        "min_confidence": {"value": round(65 - risk_level * 35, 0), "unit": "%", "desc": "Min confidence to trade"},
        "leverage_cap": {"value": round(1.0 + risk_level * 1.0, 2), "unit": "x", "desc": "Max leverage"},
    }
    if custom:
        for k, v in custom.items():
            if k in params and v is not None:
                params[k]["value"] = v
                params[k]["custom"] = True
    return params


# ============== STOCK DATA ==============
@st.cache_data(ttl=300)
def fetch_stock(ticker: str) -> dict:
    ts = datetime.now()
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        info = stock.info
        if len(hist) >= 1:
            price = float(hist['Close'].iloc[-1])
            prev = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else price
            return {
                "valid": True, "ticker": ticker, "price": price,
                "change": price - prev, "change_pct": ((price - prev) / prev * 100) if prev else 0,
                "name": info.get("shortName", ticker), "sector": info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", 0), "pe_ratio": info.get("trailingPE") or 0,
                "beta": info.get("beta") or 1.0, "high_52w": info.get("fiftyTwoWeekHigh") or 0,
                "low_52w": info.get("fiftyTwoWeekLow") or 0, "timestamp": ts
            }
    except:
        pass
    return {"valid": False, "ticker": ticker, "price": 0, "change": 0, "change_pct": 0,
            "name": ticker, "sector": "Unknown", "timestamp": ts}


@st.cache_data(ttl=300)
def fetch_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        import yfinance as yf
        return yf.Ticker(ticker).history(period=period)
    except:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_news(ticker: str) -> List[dict]:
    """Fetch recent news for a ticker using yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        news = stock.news
        results = []
        for item in news[:10]:  # Limit to 10 articles
            results.append({
                "title": item.get("title", ""),
                "publisher": item.get("publisher", "Unknown"),
                "link": item.get("link", ""),
                "timestamp": datetime.fromtimestamp(item.get("providerPublishTime", 0)) if item.get("providerPublishTime") else None,
                "type": item.get("type", "article")
            })
        return results
    except:
        return []


# ============== ALLOCATION PARAMETERS ==============
def get_allocation_params(mode_key: str, custom: dict = None) -> dict:
    """Get allocation parameters for a given mode with optional custom overrides."""
    mode = ALLOCATION_MODES.get(mode_key, ALLOCATION_MODES["max_deploy"])
    params = {
        "target_deploy_pct": {"value": (mode.get("target_pct") or 0.5) * 100, "unit": "%", "desc": "Target deployment"},
        "position_cap_pct": {"value": 35.0, "unit": "%", "desc": "Max single position"},
        "scale_caps": {"value": mode_key == "max_deploy", "unit": "", "desc": "Scale caps with # tickers"},
        "min_position_pct": {"value": 2.0, "unit": "%", "desc": "Min position size"},
    }
    if custom:
        for k, v in custom.items():
            if k in params and v is not None:
                params[k]["value"] = v
                params[k]["custom"] = True
    return params


# ============== ANALYSIS ENGINE ==============
def run_analysis(tickers: List[str], analysts: List[str], risk_level: float, capital: float,
                 holdings: Dict[str, int], mode_key: str, allow_fractional: bool = False,
                 custom_params: dict = None) -> dict:

    sorted_analysts = sorted(analysts)
    seed_str = f"{sorted(tickers)}{sorted_analysts}{risk_level:.2f}{capital}{mode_key}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)

    risk_params = get_risk_params(risk_level, custom_params)
    mode = ALLOCATION_MODES[mode_key]
    timestamp = datetime.now()

    audit = {"inputs": {"tickers": tickers, "analysts": sorted_analysts, "capital": capital,
                        "mode": mode["name"], "risk_level": risk_level}, "steps": []}

    # PHASE 1: SIGNALS
    ticker_results = {}
    for ticker in tickers:
        np.random.seed(seed)
        stock = fetch_stock(ticker)
        signals = []

        for analyst_key in sorted_analysts:
            if analyst_key not in ALL_ANALYSTS:
                continue
            info = ALL_ANALYSTS[analyst_key]
            sig_seed = int(hashlib.md5(f"{analyst_key}{ticker}{seed}".encode()).hexdigest()[:8], 16)
            np.random.seed(sig_seed)

            score = np.random.uniform(-1, 1) + info.get("bias", 0)
            if score > 0.2:
                signal, confidence = "BULLISH", 50 + score * 40
            elif score < -0.2:
                signal, confidence = "BEARISH", 50 + abs(score) * 40
            else:
                signal, confidence = "NEUTRAL", 40 + np.random.uniform(0, 20)

            signals.append({
                "analyst": info["name"], "analyst_key": analyst_key, "category": info["category"],
                "signal": signal, "confidence": min(95, max(30, confidence)), "score": score,
                "thesis": info.get("thesis", ""), "drivers": info.get("drivers", ""),
                "horizon": info.get("horizon", ""), "risks": info.get("risks", "")
            })

        bullish = sum(1 for s in signals if s["signal"] == "BULLISH")
        bearish = sum(1 for s in signals if s["signal"] == "BEARISH")
        neutral = len(signals) - bullish - bearish
        total = len(signals)
        avg_conf = np.mean([s["confidence"] for s in signals]) if signals else 50
        min_conf = risk_params["min_confidence"]["value"]

        if total == 0:
            action, reason = "HOLD", "No analysts selected"
        elif bullish > bearish and bullish >= neutral and avg_conf >= min_conf:
            action = "BUY"
            reason = f"Bullish consensus ({bullish}/{total}) at {avg_conf:.0f}% >= {min_conf:.0f}% threshold"
        elif bearish > bullish and bearish >= neutral and avg_conf >= min_conf:
            action = "SHORT"
            reason = f"Bearish consensus ({bearish}/{total}) at {avg_conf:.0f}% >= {min_conf:.0f}% threshold"
        elif avg_conf < min_conf:
            action, reason = "HOLD", f"Confidence {avg_conf:.0f}% below {min_conf:.0f}% threshold"
        else:
            action, reason = "HOLD", f"No consensus ({bullish}B/{neutral}N/{bearish}Be)"

        ticker_results[ticker] = {
            "stock": stock, "signals": signals, "bullish": bullish, "bearish": bearish,
            "neutral": neutral, "total": total, "avg_confidence": avg_conf,
            "action": action, "reason": reason, "holdings": holdings.get(ticker, 0)
        }

    for t, r in ticker_results.items():
        audit["steps"].append(("signal", f"{t} → {r['action']} ({r['bullish']}B/{r['neutral']}N/{r['bearish']}Be, {r['avg_confidence']:.0f}% conf)"))

    # PHASE 2: ALLOCATION
    actionable = {t: r for t, r in ticker_results.items() if r["action"] != "HOLD" and r["stock"]["valid"]}
    n_actionable = len(actionable)
    positions = {}
    hold_tickers = {}
    cap_blocked = 0
    rounding_remainder = 0

    if n_actionable == 0:
        audit["steps"].append(("info", "No actionable tickers - 100% cash"))
        for t, r in ticker_results.items():
            hold_tickers[t] = r["reason"]
    else:
        base_cap = risk_params["max_position_pct"]["value"] / 100
        target_pct = mode.get("target_pct") or 0.5

        if mode_key == "max_deploy":
            effective_cap = min(target_pct / n_actionable, 0.95)
            if effective_cap > base_cap:
                audit["steps"].append(("info", f"Position cap scaled: {base_cap:.1%} → {effective_cap:.1%} (for {n_actionable} ticker(s))"))
        else:
            effective_cap = base_cap

        target_total = capital * target_pct
        audit["steps"].append(("info", f"Target: deploy {target_pct:.0%} of {capital:,.0f} = {target_total:,.0f}"))

        if mode_key != "confidence_weighted":
            per_pos_budget = target_total / n_actionable
            audit["steps"].append(("info", f"Per-position budget: {per_pos_budget:,.0f}"))

        for ticker, result in actionable.items():
            stock = result["stock"]
            price = stock["price"]

            if mode_key == "confidence_weighted":
                total_conf = sum(r["avg_confidence"] for r in actionable.values())
                weight = result["avg_confidence"] / total_conf if total_conf > 0 else 1/n_actionable
                budget = capital * effective_cap * weight * (result["avg_confidence"] / 100)
            else:
                budget = per_pos_budget

            max_budget = capital * effective_cap
            capped = budget > max_budget
            if capped:
                blocked = budget - max_budget
                cap_blocked += blocked
                budget = max_budget
                audit["steps"].append(("warning", f"{ticker}: capped at {max_budget:,.0f} ({blocked:,.0f} blocked)"))

            if allow_fractional:
                shares = round(budget / price, 2)
                actual = shares * price
                remainder = budget - actual
            else:
                shares = int(budget / price)
                actual = shares * price
                remainder = budget - actual

            rounding_remainder += remainder

            sl_pct = risk_params["stop_loss_pct"]["value"]
            tp_pct = risk_params["take_profit_pct"]["value"]

            if result["action"] == "BUY":
                sl_price, tp_price = price * (1 - sl_pct / 100), price * (1 + tp_pct / 100)
                sl_dir, tp_dir = "below", "above"
            else:
                sl_price, tp_price = price * (1 + sl_pct / 100), price * (1 - tp_pct / 100)
                sl_dir, tp_dir = "above", "below"

            current = result["holdings"]
            delta = shares - current if result["action"] == "BUY" else -shares - current

            positions[ticker] = {
                "action": result["action"], "shares": shares, "price": price,
                "notional": actual, "pct": (actual / capital * 100) if capital else 0,
                "capped": capped, "sl_price": sl_price, "sl_pct": sl_pct, "sl_dir": sl_dir,
                "tp_price": tp_price, "tp_pct": tp_pct, "tp_dir": tp_dir,
                "current": current, "delta": delta, "confidence": result["avg_confidence"]
            }

            shares_fmt = f"{shares:.2f}" if allow_fractional else f"{shares:,}"
            audit["steps"].append(("alloc", f"{ticker}: {result['action']} {shares_fmt} shares @ {price:.2f} = {actual:,.0f} ({actual/capital*100:.1f}%)"))

        for t, r in ticker_results.items():
            if t not in positions:
                hold_tickers[t] = r["reason"]

    # Calculate existing holdings value
    existing_long = 0
    existing_short = 0
    for ticker, result in ticker_results.items():
        current_shares = result["holdings"]
        if current_shares > 0 and result["stock"]["valid"]:
            existing_long += current_shares * result["stock"]["price"]
        elif current_shares < 0 and result["stock"]["valid"]:
            existing_short += abs(current_shares) * result["stock"]["price"]

    long_exp = sum(p["notional"] for p in positions.values() if p["action"] == "BUY")
    short_exp = sum(p["notional"] for p in positions.values() if p["action"] == "SHORT")
    gross = long_exp + short_exp
    cash = capital - gross

    # Calculate net trades needed (delta from current to target)
    total_buys = sum(p["delta"] * p["price"] for p in positions.values() if p["delta"] > 0)
    total_sells = sum(abs(p["delta"]) * p["price"] for p in positions.values() if p["delta"] < 0)

    audit["steps"].append(("result", f"Result: {gross:,.0f} deployed ({gross/capital*100:.1f}%), {cash:,.0f} cash ({cash/capital*100:.1f}%)"))
    if existing_long > 0 or existing_short > 0:
        audit["steps"].append(("info", f"Existing holdings: ${existing_long:,.0f} long, ${existing_short:,.0f} short"))
    if total_buys > 0 or total_sells > 0:
        audit["steps"].append(("info", f"Net trades: ${total_buys:,.0f} to buy, ${total_sells:,.0f} to sell"))
    if cap_blocked > 0:
        audit["steps"].append(("warning", f"Cap blocked: {cap_blocked:,.0f}"))
    if rounding_remainder > 1:
        audit["steps"].append(("info", f"Rounding remainder: {rounding_remainder:,.0f}"))

    return {
        "timestamp": timestamp,
        "config": {"tickers": tickers, "analysts": sorted_analysts, "analyst_count": len(sorted_analysts),
                   "capital": capital, "mode": mode["name"], "risk_level": risk_level, "holdings": holdings},
        "risk_params": risk_params,
        "ticker_results": ticker_results,
        "positions": positions,
        "hold_tickers": hold_tickers,
        "audit": audit,
        "summary": {
            "capital": capital, "deployed": gross, "deployed_pct": (gross/capital*100) if capital else 0,
            "cash": cash, "cash_pct": (cash/capital*100) if capital else 0,
            "long": long_exp, "short": short_exp, "gross": gross, "net": long_exp - short_exp,
            "cap_blocked": cap_blocked, "rounding": rounding_remainder,
            "positions_count": len(positions), "hold_count": len(hold_tickers),
            "existing_long": existing_long, "existing_short": existing_short,
            "total_buys": total_buys, "total_sells": total_sells
        }
    }


# ============== SESSION STATE ==============
if "result" not in st.session_state:
    st.session_state.result = None
if "custom_params" not in st.session_state:
    st.session_state.custom_params = {}
if "use_custom" not in st.session_state:
    st.session_state.use_custom = False
if "alloc_custom_params" not in st.session_state:
    st.session_state.alloc_custom_params = {}
if "use_alloc_custom" not in st.session_state:
    st.session_state.use_alloc_custom = False
if "chart_period" not in st.session_state:
    st.session_state.chart_period = "1y"
if "risk_level" not in st.session_state:
    st.session_state.risk_level = 0.5
# Initialize checkbox states for all analysts (default: all selected)
for key in ALL_ANALYST_KEYS:
    if f"chk_{key}" not in st.session_state:
        st.session_state[f"chk_{key}"] = True

# Track which analyst category expanders should stay open
for cat in ANALYST_CATEGORIES.keys():
    cat_id = cat.replace(" ", "_")
    if f"expand_{cat_id}" not in st.session_state:
        st.session_state[f"expand_{cat_id}"] = False


def select_all_analysts():
    """Set all analyst checkboxes to True."""
    for key in ALL_ANALYST_KEYS:
        st.session_state[f"chk_{key}"] = True


def clear_all_analysts():
    """Set all analyst checkboxes to False."""
    for key in ALL_ANALYST_KEYS:
        st.session_state[f"chk_{key}"] = False


def make_category_callback(cat_id: str):
    """Factory to create a callback that keeps a category expander open."""
    def callback():
        st.session_state[f"expand_{cat_id}"] = True
    return callback


def get_selected_analysts():
    """Get list of currently selected analyst keys based on checkbox states."""
    return [key for key in ALL_ANALYST_KEYS if st.session_state.get(f"chk_{key}", False)]


# ============== HEADER ==============
st.write("# 📊 AI Hedge Fund Terminal")
st.caption("v4.8 | Yahoo Finance (15-20 min delayed)")

# ============== TABS ==============
tab_signals, tab_portfolio, tab_trades, tab_analysts, tab_securities, tab_settings = st.tabs([
    "📈 Signals", "💼 Portfolio", "📋 Trades", "🧠 Analysts", "🔍 Securities", "⚙️ Settings"
])


# ============== SIGNALS TAB ==============
with tab_signals:
    col_config, col_results = st.columns([1, 2])

    with col_config:
        st.subheader("Configuration")

        ticker_input = st.text_input("Stock Tickers", value="AAPL, MSFT, NVDA, GOOGL",
                                      help="Comma-separated symbols", key="ticker_input")
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        st.caption(f"{len(tickers)} ticker(s)")

        capital = st.number_input("Investment Capital ($)", min_value=1000, value=100000, step=10000, key="capital_input")

        with st.expander("Current Holdings", expanded=False):
            holdings_text = st.text_area("TICKER:SHARES per line", placeholder="AAPL:50\nMSFT:30", height=80, key="holdings_input")
            holdings = {}
            for line in holdings_text.strip().split("\n"):
                if ":" in line:
                    try:
                        t, s = line.split(":")
                        holdings[t.strip().upper()] = int(s.strip())
                    except:
                        pass

        st.divider()

        mode_key = st.selectbox("Allocation Mode", options=list(ALLOCATION_MODES.keys()),
                                 format_func=lambda x: ALLOCATION_MODES[x]["name"], key="mode_select")
        st.caption(ALLOCATION_MODES[mode_key]["desc"])

        allow_fractional = st.checkbox("Allow fractional shares", help="Paper trading only", key="fractional_check")

        st.divider()

        st.write("**Risk Settings**")
        risk_level = st.slider("Risk Level", 0.0, 1.0, st.session_state.risk_level, 0.05, key="risk_slider")
        st.session_state.risk_level = risk_level

        risk_label = "Conservative" if risk_level < 0.35 else "Aggressive" if risk_level > 0.65 else "Moderate"
        st.caption(f"{risk_label} ({risk_level:.0%})")

        effective_params = get_risk_params(st.session_state.risk_level,
                                            st.session_state.custom_params if st.session_state.use_custom else None)
        if st.session_state.use_custom:
            st.warning("⚠️ Custom parameters active")

        with st.expander("Active Risk Parameters", expanded=False):
            for key, param in effective_params.items():
                status = " ✏️" if param.get("custom") else ""
                st.write(f"**{param['desc']}:** {param['value']}{param['unit']}{status}")

        st.divider()

        # ANALYSTS - CHECKBOX UI WITH PROPER SELECT ALL/CLEAR ALL SYNC
        st.write("**AI Analysts**")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", key="btn_select_all", use_container_width=True, on_click=select_all_analysts):
                pass  # on_click handles it
        with col2:
            if st.button("Clear All", key="btn_clear_all", use_container_width=True, on_click=clear_all_analysts):
                pass  # on_click handles it

        # Create expanders for each category
        for cat, analysts in ANALYST_CATEGORIES.items():
            cat_count = sum(1 for k in analysts if st.session_state.get(f"chk_{k}", False))
            cat_id = cat.replace(" ", "_")
            is_expanded = st.session_state.get(f"expand_{cat_id}", False)

            with st.expander(f"{cat} ({cat_count}/{len(analysts)})", expanded=is_expanded):
                for key, info in analysts.items():
                    # Checkbox with on_change to keep this category's expander open
                    st.checkbox(info["name"], key=f"chk_{key}", help=info["desc"],
                               on_change=make_category_callback(cat_id))

        selected_analysts = get_selected_analysts()
        selected_count = len(selected_analysts)
        st.write(f"**{selected_count}/{len(ALL_ANALYST_KEYS)}** analysts selected")

        st.divider()

        can_run = len(tickers) > 0 and selected_count > 0

        if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True, disabled=not can_run, key="run_btn"):
            custom = st.session_state.custom_params if st.session_state.use_custom else None
            st.session_state.result = run_analysis(
                tickers=tickers,
                analysts=selected_analysts,
                risk_level=st.session_state.risk_level,
                capital=capital,
                holdings=holdings,
                mode_key=mode_key,
                allow_fractional=allow_fractional,
                custom_params=custom
            )
            # Reset all analyst category expanders after running
            for cat in ANALYST_CATEGORIES.keys():
                st.session_state[f"expand_{cat.replace(' ', '_')}"] = False
            st.rerun()

        if not can_run:
            if len(tickers) == 0:
                st.warning("Enter at least one ticker")
            if selected_count == 0:
                st.warning("Select at least one analyst")

    # ===== RESULTS =====
    with col_results:
        if st.session_state.result:
            r = st.session_state.result
            s = r["summary"]

            st.subheader("Results")
            st.caption(f"Mode: {r['config']['mode']} | {r['config']['analyst_count']} analysts | {r['timestamp'].strftime('%H:%M:%S')}")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write("**Capital**")
                st.write(f"### ${s['capital']:,.0f}")
            with col2:
                st.write("**Deployed**")
                st.write(f"### ${s['deployed']:,.0f}")
                st.caption(f"{s['deployed_pct']:.1f}%")
            with col3:
                st.write("**Cash**")
                st.write(f"### ${s['cash']:,.0f}")
                st.caption(f"{s['cash_pct']:.1f}%")
            with col4:
                max_loss = sum(p['notional'] * p['sl_pct']/100 for p in r['positions'].values())
                st.write("**Max Loss**")
                st.write(f"### ${max_loss:,.0f}")

            st.divider()

            st.write("### Allocation Audit Trail")
            audit = r["audit"]
            st.write(f"**Inputs:** {len(audit['inputs']['tickers'])} tickers, {len(audit['inputs']['analysts'])} analysts, "
                     f"${audit['inputs']['capital']:,.0f} capital, {audit['inputs']['mode']}, {audit['inputs']['risk_level']:.0%} risk")

            for step_type, step_text in audit["steps"]:
                if step_type == "signal":
                    st.write(f"🎯 {step_text}")
                elif step_type == "warning":
                    st.write(f"⚠️ {step_text}")
                elif step_type == "result":
                    st.success(step_text)
                elif step_type == "info":
                    st.info(step_text)
                else:
                    st.write(f"→ {step_text}")

            st.divider()

            st.write("### Exposure")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write("**Target Long**")
                st.write(f"### ${s['long']:,.0f}")
            with col2:
                st.write("**Target Short**")
                st.write(f"### ${s['short']:,.0f}")
            with col3:
                st.write("**Gross**")
                st.write(f"### ${s['gross']:,.0f}")
            with col4:
                st.write("**Net**")
                st.write(f"### ${s['net']:,.0f}")

            # Show existing holdings impact
            if s.get('existing_long', 0) > 0 or s.get('existing_short', 0) > 0:
                st.write("#### Current Holdings")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write("**Existing Long**")
                    st.write(f"${s.get('existing_long', 0):,.0f}")
                with col2:
                    st.write("**Existing Short**")
                    st.write(f"${s.get('existing_short', 0):,.0f}")
                with col3:
                    st.write("**To Buy**")
                    st.write(f"${s.get('total_buys', 0):,.0f}")
                with col4:
                    st.write("**To Sell**")
                    st.write(f"${s.get('total_sells', 0):,.0f}")

            if s['short'] > 0:
                st.info(f"Short margin required: ${s['short'] * 0.5:,.0f} (50% Reg T)")

            st.divider()

            st.write("### Recommendations")
            for ticker, tr in r["ticker_results"].items():
                stock = tr["stock"]
                pos = r["positions"].get(ticker)

                col1, col2 = st.columns([4, 1])
                with col1:
                    if stock["valid"]:
                        chg = "▲" if stock["change"] >= 0 else "▼"
                        st.write(f"**{ticker}** — ${stock['price']:.2f} {chg}{abs(stock['change_pct']):.2f}%")
                        st.caption(f"{stock['name']} | {stock['sector']}")
                    else:
                        st.write(f"**{ticker}** — Price unavailable")
                with col2:
                    if tr["action"] == "BUY":
                        st.success(f"📈 BUY")
                    elif tr["action"] == "SHORT":
                        st.error(f"📉 SHORT")
                    else:
                        st.warning(f"⏸️ HOLD")

                st.caption(f"**Reason:** {tr['reason']}")

                if pos:
                    col1, col2, col3, col4 = st.columns(4)
                    shares_fmt = f"{pos['shares']:.2f}" if isinstance(pos['shares'], float) and pos['shares'] != int(pos['shares']) else f"{int(pos['shares']):,}"
                    with col1:
                        st.write("**Target Shares**")
                        st.write(f"### {shares_fmt}")
                        if pos['current'] != 0:
                            st.caption(f"Current: {pos['current']:,}")
                    with col2:
                        st.write("**Notional**")
                        st.write(f"### ${pos['notional']:,.0f}")
                        if pos['delta'] != 0:
                            delta_val = pos['delta'] * pos['price']
                            st.caption(f"Delta: {'+' if pos['delta'] > 0 else ''}{pos['delta']:,} (${delta_val:+,.0f})")
                    with col3:
                        st.write("**Stop Loss**")
                        st.write(f"### ${pos['sl_price']:.2f}")
                    with col4:
                        st.write("**Take Profit**")
                        st.write(f"### ${pos['tp_price']:.2f}")

                # Show all analyst signals - fixed to always display complete list
                with st.expander(f"View {tr['total']} analyst signals"):
                    sig_data = []
                    for sig in sorted(tr["signals"], key=lambda x: (-x["confidence"], x["analyst"])):
                        sig_data.append({
                            "Analyst": sig["analyst"],
                            "Category": sig["category"],
                            "Signal": sig["signal"],
                            "Confidence": f"{sig['confidence']:.0f}%"
                        })
                    if sig_data:
                        st.dataframe(pd.DataFrame(sig_data), hide_index=True, use_container_width=True)
                    else:
                        st.write("No analyst signals generated.")

                st.divider()

            st.write("### Export")
            if r["positions"]:
                csv_rows = ["Ticker,Action,Shares,Entry,Notional,StopLoss,TakeProfit,Confidence"]
                for t, p in r["positions"].items():
                    shares_str = f"{p['shares']:.2f}" if isinstance(p['shares'], float) else str(p['shares'])
                    csv_rows.append(f"{t},{p['action']},{shares_str},{p['price']:.2f},{p['notional']:.0f},{p['sl_price']:.2f},{p['tp_price']:.2f},{p['confidence']:.0f}")
                csv_data = "\n".join(csv_rows)
            else:
                csv_data = "No trades"

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 CSV", csv_data, f"trades_{r['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
                                   "text/csv", key="dl_csv_signals", use_container_width=True)
            with col2:
                text_lines = [f"{p['action']} {p['shares']} {t} @ ${p['price']:.2f}" for t, p in r["positions"].items()]
                st.download_button("📋 Text", "\n".join(text_lines) if text_lines else "No trades",
                                   f"trades_{r['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                                   "text/plain", key="dl_txt_signals", use_container_width=True)

        else:
            st.write("""
            ### Getting Started

            1. Enter ticker symbols (comma-separated)
            2. Set investment capital
            3. Choose allocation mode
            4. Adjust risk level
            5. Select AI analysts
            6. Click **RUN ANALYSIS**
            """)


# ============== PORTFOLIO TAB ==============
with tab_portfolio:
    st.subheader("Portfolio Overview")

    if st.session_state.result:
        r = st.session_state.result
        s = r["summary"]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("**Positions**")
            st.write(f"### {s['positions_count']}")
        with col2:
            st.write("**HOLD**")
            st.write(f"### {s['hold_count']}")
        with col3:
            st.write("**Deployed**")
            st.write(f"### {s['deployed_pct']:.1f}%")
        with col4:
            st.write("**Cash**")
            st.write(f"### {s['cash_pct']:.1f}%")

        st.divider()

        if r["positions"]:
            st.write("### Positions")
            pos_data = []
            for t, p in r["positions"].items():
                shares_fmt = f"{p['shares']:.2f}" if isinstance(p['shares'], float) and p['shares'] != int(p['shares']) else int(p['shares'])
                pos_data.append({
                    "Ticker": t, "Action": p["action"], "Shares": shares_fmt,
                    "Entry": f"${p['price']:.2f}", "Notional": f"${p['notional']:,.0f}",
                    "% Port": f"{p['pct']:.1f}%", "Stop": f"${p['sl_price']:.2f}",
                    "Target": f"${p['tp_price']:.2f}", "Conf": f"{p['confidence']:.0f}%"
                })
            st.dataframe(pd.DataFrame(pos_data), hide_index=True, use_container_width=True)

        if r["hold_tickers"]:
            st.write("### Not Trading (HOLD)")
            for t, reason in r["hold_tickers"].items():
                st.write(f"- **{t}**: {reason}")

        st.divider()
        st.write("### Cash Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Undeployed**")
            st.write(f"### ${s['cash']:,.0f}")
        with col2:
            st.write("**Rounding**")
            st.write(f"### ${s['rounding']:,.0f}")
        with col3:
            st.write("**Cap Blocked**")
            st.write(f"### ${s['cap_blocked']:,.0f}")
    else:
        st.info("Run analysis from Signals tab first.")


# ============== TRADES TAB ==============
with tab_trades:
    st.subheader("Trade Instructions")

    if st.session_state.result:
        r = st.session_state.result

        if r["positions"]:
            trade_data = []
            for t, p in r["positions"].items():
                shares_fmt = f"{p['shares']:.2f}" if isinstance(p['shares'], float) and p['shares'] != int(p['shares']) else int(p['shares'])
                delta = p["delta"]
                delta_str = "New" if p["current"] == 0 else f"{delta:+,}" if isinstance(delta, int) else f"{delta:+.2f}"
                trade_data.append({
                    "Ticker": t, "Action": p["action"], "Shares": shares_fmt,
                    "Entry": f"${p['price']:.2f}", "Notional": f"${p['notional']:,.0f}",
                    "Stop": f"${p['sl_price']:.2f}", "Target": f"${p['tp_price']:.2f}",
                    "Delta": delta_str
                })
            st.dataframe(pd.DataFrame(trade_data), hide_index=True, use_container_width=True)

            st.divider()
            with st.expander("📝 Execution Notes", expanded=True):
                st.write("""
                - **Data delay**: Prices are 15-20 min delayed. Use live quotes.
                - **Order type**: Consider limit orders near entry prices.
                - **Stop losses**: Set immediately after entry.
                - **Margin**: Shorts require 50% initial margin (Reg T).
                """)

            st.divider()
            csv_rows = ["Ticker,Action,Shares,Entry,Notional,Stop,Target,Delta"]
            for t, p in r["positions"].items():
                shares_str = f"{p['shares']:.2f}" if isinstance(p['shares'], float) else str(p['shares'])
                delta_str = "New" if p["current"] == 0 else str(p["delta"])
                csv_rows.append(f"{t},{p['action']},{shares_str},{p['price']:.2f},{p['notional']:.0f},{p['sl_price']:.2f},{p['tp_price']:.2f},{delta_str}")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button("📥 CSV", "\n".join(csv_rows), "trades.csv", "text/csv", key="dl_csv_trades", use_container_width=True)
            with col2:
                text_lines = [f"{p['action']} {p['shares']} {t} @ ${p['price']:.2f}" for t, p in r["positions"].items()]
                st.download_button("📋 Text", "\n".join(text_lines), "trades.txt", "text/plain", key="dl_txt_trades", use_container_width=True)
        else:
            st.info("No trades. All positions are HOLD.")
    else:
        st.info("Run analysis first.")


# ============== ANALYSTS TAB ==============
with tab_analysts:
    st.subheader("AI Analysts")

    if st.session_state.result:
        r = st.session_state.result
        selected = r["config"]["analysts"]

        st.write(f"### Selected Analysts ({len(selected)})")

        for analyst_key in selected:
            if analyst_key not in ALL_ANALYSTS:
                continue
            info = ALL_ANALYSTS[analyst_key]

            with st.expander(f"**{info['name']}** ({info['category']})"):
                st.write(f"**Thesis:** {info.get('thesis', 'N/A')}")
                st.write(f"**Key Drivers:** {info.get('drivers', 'N/A')}")
                st.write(f"**Time Horizon:** {info.get('horizon', 'N/A')}")
                st.write(f"**Risks:** {info.get('risks', 'N/A')}")
                st.write(f"**Bias:** {info.get('bias', 0):+.2f}")

                st.write("**Signals in this run:**")
                sig_data = []
                for ticker, tr in r["ticker_results"].items():
                    for sig in tr["signals"]:
                        if sig["analyst_key"] == analyst_key:
                            sig_data.append({"Ticker": ticker, "Signal": sig["signal"], "Confidence": f"{sig['confidence']:.0f}%"})
                if sig_data:
                    st.dataframe(pd.DataFrame(sig_data), hide_index=True, use_container_width=True)

        st.divider()

        st.write("### Consensus Breakdown")
        for ticker, tr in r["ticker_results"].items():
            with st.expander(f"**{ticker}** → {tr['action']} ({tr['avg_confidence']:.0f}% confidence)"):
                st.write(f"**Final Decision:** {tr['action']}")
                st.write(f"**Reason:** {tr['reason']}")
                st.write(f"**Vote:** {tr['bullish']} Bullish / {tr['neutral']} Neutral / {tr['bearish']} Bearish")

                for signal_type, label, emoji in [("BULLISH", "Bullish", "📈"), ("BEARISH", "Bearish", "📉"), ("NEUTRAL", "Neutral", "⏸️")]:
                    sigs = [s for s in tr["signals"] if s["signal"] == signal_type]
                    if sigs:
                        st.write(f"---")
                        st.write(f"**{emoji} {label} Arguments ({len(sigs)}):**")
                        for s in sorted(sigs, key=lambda x: -x["confidence"]):
                            st.write(f"**{s['analyst']}** ({s['confidence']:.0f}%)")
                            st.write(f"- Thesis: {s['thesis']}")
                            st.write(f"- Drivers: {s['drivers']}")
    else:
        st.write("### All Available Analysts")
        for cat, analysts in ANALYST_CATEGORIES.items():
            st.write(f"#### {cat}")
            for key, info in analysts.items():
                selected = "✅" if st.session_state.get(f"chk_{key}", False) else "⬜"
                st.write(f"{selected} **{info['name']}** - {info['desc']}")


# ============== SECURITIES TAB ==============
with tab_securities:
    st.subheader("Securities Lookup")

    ticker = st.text_input("Enter ticker symbol", value="AAPL", placeholder="AAPL", key="securities_ticker")

    if ticker:
        ticker = ticker.upper()
        with st.spinner(f"Fetching {ticker}..."):
            stock = fetch_stock(ticker)

        if stock["valid"]:
            st.caption(f"Yahoo Finance | 15-20 min delayed | {stock['timestamp'].strftime('%H:%M:%S')}")

            col1, col2 = st.columns([2, 1])
            with col1:
                chg = "▲" if stock["change"] >= 0 else "▼"
                st.write(f"## {stock['ticker']}")
                st.write(f"*{stock['name']}*")
                st.write(f"### ${stock['price']:.2f}")
                color = "green" if stock["change"] >= 0 else "red"
                st.write(f":{color}[{chg} ${abs(stock['change']):.2f} ({abs(stock['change_pct']):.2f}%)]")
            with col2:
                st.write("**Sector**")
                st.write(stock["sector"])
                if stock["market_cap"]:
                    cap = stock["market_cap"]
                    cap_str = f"${cap/1e12:.2f}T" if cap >= 1e12 else f"${cap/1e9:.1f}B" if cap >= 1e9 else f"${cap/1e6:.0f}M"
                    st.write("**Market Cap**")
                    st.write(cap_str)

            st.divider()
            st.write("### Key Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.write("**P/E Ratio**")
                st.write(f"{stock['pe_ratio']:.1f}" if stock['pe_ratio'] else "N/A")
            with col2:
                st.write("**Beta**")
                st.write(f"{stock['beta']:.2f}" if stock['beta'] else "N/A")
            with col3:
                st.write("**52W High**")
                st.write(f"${stock['high_52w']:.2f}" if stock['high_52w'] else "N/A")
            with col4:
                st.write("**52W Low**")
                st.write(f"${stock['low_52w']:.2f}" if stock['low_52w'] else "N/A")

            st.divider()
            st.write("### Price Chart")

            periods = {"1M": "1mo", "3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "5Y": "5y", "MAX": "max"}
            cols = st.columns(len(periods))
            for i, (label, period) in enumerate(periods.items()):
                with cols[i]:
                    btn_type = "primary" if st.session_state.chart_period == period else "secondary"
                    if st.button(label, key=f"chart_btn_{label}", use_container_width=True, type=btn_type):
                        st.session_state.chart_period = period
                        st.rerun()

            hist = fetch_history(ticker, st.session_state.chart_period)
            if len(hist) > 0:
                st.line_chart(hist["Close"])
                if len(hist) > 1:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write("**Period High**")
                        st.write(f"${hist['Close'].max():.2f}")
                    with col2:
                        st.write("**Period Low**")
                        st.write(f"${hist['Close'].min():.2f}")
                    with col3:
                        change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)
                        st.write("**Period Change**")
                        st.write(f"{change:+.1f}%")
                    with col4:
                        st.write("**Volatility**")
                        st.write(f"{hist['Close'].std():.2f}")
            else:
                st.warning("Chart data unavailable")

            # News Section
            st.divider()
            st.write("### Recent News")
            try:
                news = fetch_news(ticker)
                if news:
                    for i, article in enumerate(news[:5]):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            if article.get("link"):
                                st.write(f"**[{article['title']}]({article['link']})**")
                            else:
                                st.write(f"**{article['title']}**")
                        with col2:
                            if article.get("timestamp"):
                                st.caption(article["timestamp"].strftime("%b %d"))
                            st.caption(article.get("publisher", ""))
                        if i < len(news[:5]) - 1:
                            st.write("---")
                else:
                    st.info("No recent news available for this ticker.")
            except Exception:
                st.info("News data temporarily unavailable.")

            if st.session_state.result and ticker in st.session_state.result["ticker_results"]:
                st.divider()
                st.write("### From Current Analysis")
                tr = st.session_state.result["ticker_results"][ticker]
                col1, col2 = st.columns(2)
                with col1:
                    if tr["action"] == "BUY":
                        st.success(f"📈 {tr['action']}")
                    elif tr["action"] == "SHORT":
                        st.error(f"📉 {tr['action']}")
                    else:
                        st.warning(f"⏸️ {tr['action']}")
                with col2:
                    st.write("**Confidence**")
                    st.write(f"### {tr['avg_confidence']:.0f}%")
                st.caption(f"**Reason:** {tr['reason']}")
        else:
            st.error(f"Could not fetch data for {ticker}")


# ============== SETTINGS TAB ==============
with tab_settings:
    st.subheader("Settings")

    tab_risk, tab_allocation = st.tabs(["⚠️ Risk Parameters", "📊 Allocation Parameters"])

    # ===== RISK TAB =====
    with tab_risk:
        st.write("### Risk Level Presets")
        preset_data = []
        for level in [0.0, 0.25, 0.5, 0.75, 1.0]:
            params = get_risk_params(level)
            label = "Very Conservative" if level == 0 else "Conservative" if level == 0.25 else "Moderate" if level == 0.5 else "Aggressive" if level == 0.75 else "Very Aggressive"
            preset_data.append({
                "Risk": f"{level:.0%} ({label})",
                "Max Position": f"{params['max_position_pct']['value']}%",
                "Stop Loss": f"{params['stop_loss_pct']['value']}%",
                "Take Profit": f"{params['take_profit_pct']['value']}%",
                "Min Confidence": f"{params['min_confidence']['value']}%",
                "Leverage": f"{params['leverage_cap']['value']}x"
            })
        st.dataframe(pd.DataFrame(preset_data), hide_index=True, use_container_width=True)

        st.divider()

        # Current Active Risk Parameters
        st.write("### Current Active Risk Parameters")
        risk_label = "Conservative" if st.session_state.risk_level < 0.35 else "Aggressive" if st.session_state.risk_level > 0.65 else "Moderate"
        st.caption(f"Risk level: {st.session_state.risk_level:.0%} ({risk_label})" + (" with custom overrides" if st.session_state.use_custom else ""))

        current_params = get_risk_params(st.session_state.risk_level, st.session_state.custom_params if st.session_state.use_custom else None)
        col1, col2 = st.columns(2)
        param_items = list(current_params.items())
        for i, (key, param) in enumerate(param_items):
            with col1 if i % 2 == 0 else col2:
                status = " ✏️" if param.get("custom") else ""
                st.write(f"**{param['desc']}:** {param['value']}{param['unit']}{status}")

        st.divider()

        # Custom Risk Parameters
        st.write("### Custom Risk Overrides")
        use_custom = st.checkbox("Enable custom risk parameters", value=st.session_state.use_custom, key="use_custom_check")
        st.session_state.use_custom = use_custom

        if use_custom:
            st.warning("⚠️ Custom parameters override preset values")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.custom_params["max_position_pct"] = st.number_input("Max Position %", 5.0, 100.0, float(st.session_state.custom_params.get("max_position_pct", 22.5)), 1.0, key="custom_max_pos")
                st.session_state.custom_params["stop_loss_pct"] = st.number_input("Stop Loss %", 1.0, 50.0, float(st.session_state.custom_params.get("stop_loss_pct", 12.5)), 0.5, key="custom_sl")
                st.session_state.custom_params["take_profit_pct"] = st.number_input("Take Profit %", 5.0, 200.0, float(st.session_state.custom_params.get("take_profit_pct", 37.5)), 1.0, key="custom_tp")
            with col2:
                st.session_state.custom_params["min_confidence"] = st.number_input("Min Confidence %", 10.0, 90.0, float(st.session_state.custom_params.get("min_confidence", 47.5)), 5.0, key="custom_conf")
                st.session_state.custom_params["leverage_cap"] = st.number_input("Leverage Cap", 1.0, 5.0, float(st.session_state.custom_params.get("leverage_cap", 1.5)), 0.1, key="custom_lev")

            if st.button("Reset Risk to Defaults", use_container_width=True, key="reset_custom"):
                st.session_state.custom_params = {}
                st.session_state.use_custom = False
                st.rerun()

    # ===== ALLOCATION TAB =====
    with tab_allocation:
        st.write("### Allocation Mode Presets")
        alloc_data = []
        for mode_key, mode_info in ALLOCATION_MODES.items():
            target = mode_info.get("target_pct")
            alloc_data.append({
                "Mode": mode_info["name"],
                "Target Deploy": f"{target*100:.0f}%" if target else "Variable",
                "Description": mode_info["desc"]
            })
        st.dataframe(pd.DataFrame(alloc_data), hide_index=True, use_container_width=True)

        st.divider()

        # Current Active Allocation Parameters
        st.write("### Current Active Allocation Parameters")
        # Get current mode from session state if available
        current_mode = "max_deploy"  # default
        if st.session_state.result:
            for mk, mv in ALLOCATION_MODES.items():
                if mv["name"] == st.session_state.result["config"]["mode"]:
                    current_mode = mk
                    break

        alloc_params = get_allocation_params(current_mode, st.session_state.alloc_custom_params if st.session_state.use_alloc_custom else None)
        st.caption(f"Mode: {ALLOCATION_MODES[current_mode]['name']}" + (" with custom overrides" if st.session_state.use_alloc_custom else ""))

        col1, col2 = st.columns(2)
        alloc_items = list(alloc_params.items())
        for i, (key, param) in enumerate(alloc_items):
            with col1 if i % 2 == 0 else col2:
                status = " ✏️" if param.get("custom") else ""
                val = param['value']
                if isinstance(val, bool):
                    val = "Yes" if val else "No"
                st.write(f"**{param['desc']}:** {val}{param['unit']}{status}")

        st.divider()

        # Custom Allocation Parameters
        st.write("### Custom Allocation Overrides")
        use_alloc_custom = st.checkbox("Enable custom allocation parameters", value=st.session_state.use_alloc_custom, key="use_alloc_custom_check")
        st.session_state.use_alloc_custom = use_alloc_custom

        if use_alloc_custom:
            st.warning("⚠️ Custom parameters override mode defaults")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.alloc_custom_params["target_deploy_pct"] = st.number_input(
                    "Target Deployment %", 10.0, 100.0,
                    float(st.session_state.alloc_custom_params.get("target_deploy_pct", 95.0)), 5.0,
                    key="custom_target_deploy", help="Percentage of capital to deploy"
                )
                st.session_state.alloc_custom_params["position_cap_pct"] = st.number_input(
                    "Max Single Position %", 5.0, 100.0,
                    float(st.session_state.alloc_custom_params.get("position_cap_pct", 35.0)), 5.0,
                    key="custom_pos_cap", help="Maximum allocation to any single position"
                )
            with col2:
                st.session_state.alloc_custom_params["min_position_pct"] = st.number_input(
                    "Min Position Size %", 0.5, 20.0,
                    float(st.session_state.alloc_custom_params.get("min_position_pct", 2.0)), 0.5,
                    key="custom_min_pos", help="Minimum position size to avoid tiny trades"
                )
                st.session_state.alloc_custom_params["scale_caps"] = st.checkbox(
                    "Scale caps with # tickers",
                    value=st.session_state.alloc_custom_params.get("scale_caps", True),
                    key="custom_scale_caps", help="Automatically adjust position caps based on number of tickers"
                )

            if st.button("Reset Allocation to Defaults", use_container_width=True, key="reset_alloc_custom"):
                st.session_state.alloc_custom_params = {}
                st.session_state.use_alloc_custom = False
                st.rerun()


# ============== FOOTER ==============
st.divider()
st.caption("AI Hedge Fund Terminal v4.8 | Educational Use Only | Not Financial Advice")
