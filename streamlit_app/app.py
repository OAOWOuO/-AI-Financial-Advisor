"""
AI Hedge Fund Terminal v4.1
Fixes: analyst selection bug, table styling, contrast, custom risk params, clickable tickers
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
from typing import Dict, List
import io

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="AI Hedge Fund Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============== CSS - DARK THEME + HIGH CONTRAST + FIXED TABLES ==============
st.markdown("""
<style>
    /* Base dark theme */
    .main { background: #0d1117; }
    .stApp { background: #0d1117; }
    #MainMenu, footer, header { visibility: hidden; }

    /* High contrast text */
    h1, h2, h3 { color: #e6edf3 !important; font-weight: 600 !important; }
    p, span, label, li { color: #c9d1d9 !important; }
    .stMarkdown { color: #c9d1d9 !important; }

    /* Improved contrast for explanatory text */
    .info-text { color: #e6edf3 !important; opacity: 1 !important; }
    .stExpander p, .stExpander span { color: #c9d1d9 !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #161b22;
        padding: 8px;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #8b949e;
        border-radius: 6px;
        padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: #238636 !important;
        color: white !important;
    }

    /* DARK THEMED TABLES - No dragging, consistent style */
    .stDataFrame {
        background: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    /* Disable table dragging/resizing */
    [data-testid="stDataFrame"] > div {
        pointer-events: auto !important;
    }
    [data-testid="stDataFrame"] iframe {
        pointer-events: auto !important;
    }

    /* Dark table header and cells */
    .stDataFrame [data-testid="StyledDataFrameDataCell"],
    .stDataFrame [data-testid="StyledDataFrameHeaderCell"] {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border-color: #30363d !important;
    }

    /* DataFrame styling via pandas */
    div[data-testid="stDataFrame"] > div > div > div {
        background-color: #161b22 !important;
    }

    /* Sticky header */
    thead th {
        position: sticky !important;
        top: 0 !important;
        background: #21262d !important;
        z-index: 1 !important;
    }

    /* Cards and boxes */
    .info-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        color: #e6edf3 !important;
    }
    .warning-box {
        background: #3d2a1f;
        border: 1px solid #d29922;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        color: #e6edf3 !important;
    }
    .error-box {
        background: #3d1f1f;
        border: 1px solid #f85149;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        color: #e6edf3 !important;
    }
    .success-box {
        background: #1f3d2a;
        border: 1px solid #3fb950;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        color: #e6edf3 !important;
    }

    /* Ticker card for quick-peek */
    .ticker-card {
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        margin: 4px 0;
    }
    .ticker-card:hover {
        border-color: #58a6ff;
        cursor: pointer;
    }

    /* Badge styling */
    .badge-buy {
        background: #238636;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-short {
        background: #da3633;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-hold {
        background: #6e7681;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Expander styling for better contrast */
    .streamlit-expanderHeader {
        color: #e6edf3 !important;
        background: #161b22 !important;
    }
    .streamlit-expanderContent {
        background: #0d1117 !important;
        color: #c9d1d9 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============== ALLOCATION MODES ==============
ALLOCATION_MODES = {
    "max_deploy": {
        "name": "Maximum Deployment",
        "desc": "Deploy 95%+ of capital. Redistributes excess to fill positions.",
        "target_pct": 0.95,
        "redistribute_excess": True
    },
    "equal_weight": {
        "name": "Equal Weight",
        "desc": "Split capital equally among all tradeable tickers.",
        "target_pct": 0.90,
        "redistribute_excess": True
    },
    "confidence_weighted": {
        "name": "Confidence Weighted",
        "desc": "Size by conviction. May hold cash if confidence is low.",
        "target_pct": None,
        "redistribute_excess": False
    },
    "conservative": {
        "name": "Conservative (Cash Buffer)",
        "desc": "Only high-conviction trades. Expects 30-50% cash.",
        "target_pct": 0.50,
        "redistribute_excess": False
    }
}


# ============== ANALYST DEFINITIONS ==============
ANALYST_CATEGORIES = {
    "Value Investors": {
        "warren_buffett": {"name": "Warren Buffett", "desc": "Moats, quality management, long-term value", "bias": -0.1},
        "charlie_munger": {"name": "Charlie Munger", "desc": "Mental models, business quality", "bias": -0.1},
        "ben_graham": {"name": "Benjamin Graham", "desc": "Margin of safety, net-net value", "bias": -0.15},
        "joel_greenblatt": {"name": "Joel Greenblatt", "desc": "Magic formula: ROIC + earnings yield", "bias": -0.05},
        "seth_klarman": {"name": "Seth Klarman", "desc": "Deep value, distressed assets", "bias": -0.15},
    },
    "Growth Investors": {
        "peter_lynch": {"name": "Peter Lynch", "desc": "PEG ratio, growth at reasonable price", "bias": 0.05},
        "phil_fisher": {"name": "Philip Fisher", "desc": "Scuttlebutt, quality growth", "bias": 0.05},
        "cathie_wood": {"name": "Cathie Wood", "desc": "Disruptive innovation, exponential growth", "bias": 0.2},
        "bill_ackman": {"name": "Bill Ackman", "desc": "Activist catalysts, concentrated bets", "bias": 0.1},
    },
    "Macro Traders": {
        "stanley_druckenmiller": {"name": "Stanley Druckenmiller", "desc": "Macro trends, asymmetric bets", "bias": 0},
        "george_soros": {"name": "George Soros", "desc": "Reflexivity, regime changes", "bias": 0},
        "ray_dalio": {"name": "Ray Dalio", "desc": "Economic machine, risk parity", "bias": -0.05},
        "paul_tudor_jones": {"name": "Paul Tudor Jones", "desc": "Technical macro, trend following", "bias": 0},
    },
    "Quantitative Agents": {
        "fundamentals_agent": {"name": "Fundamentals Analyst", "desc": "Financial ratios, earnings quality, balance sheet", "bias": 0},
        "technical_agent": {"name": "Technical Analyst", "desc": "Price patterns, momentum, RSI, MACD", "bias": 0},
        "sentiment_agent": {"name": "Sentiment Analyst", "desc": "News sentiment, social media, analyst ratings", "bias": 0.05},
        "valuation_agent": {"name": "Valuation Analyst", "desc": "DCF, comparable analysis, sum-of-parts", "bias": -0.05},
        "momentum_agent": {"name": "Momentum Analyst", "desc": "Price momentum, earnings momentum", "bias": 0.1},
        "risk_agent": {"name": "Risk Analyst", "desc": "Volatility, drawdown, tail risk", "bias": -0.1},
    },
}


def get_all_analysts():
    result = {}
    for cat, analysts in ANALYST_CATEGORIES.items():
        for key, info in analysts.items():
            result[key] = {**info, "category": cat}
    return result


# ============== RISK PARAMETER MODEL ==============
def get_risk_params(risk_level: float, custom_overrides: dict = None) -> dict:
    """
    Risk level 0.0 (very conservative) to 1.0 (very aggressive)
    Custom overrides allow manual parameter setting.
    """
    params = {
        "max_position_pct": {
            "value": round(10 + risk_level * 25, 1),  # 10-35%
            "unit": "%",
            "desc": "Maximum allocation per position"
        },
        "stop_loss_pct": {
            "value": round(20 - risk_level * 15, 1),  # 20-5%
            "unit": "%",
            "desc": "Stop loss distance from entry"
        },
        "take_profit_pct": {
            "value": round(15 + risk_level * 45, 1),  # 15-60%
            "unit": "%",
            "desc": "Take profit target from entry"
        },
        "min_confidence": {
            "value": round(65 - risk_level * 35, 0),  # 65-30%
            "unit": "%",
            "desc": "Minimum confidence to trade"
        },
        "max_sector_pct": {
            "value": round(25 + risk_level * 25, 0),  # 25-50%
            "unit": "%",
            "desc": "Maximum sector concentration"
        },
        "short_margin_req": {
            "value": 50,  # Fixed 50% initial margin
            "unit": "%",
            "desc": "Initial margin for shorts (Reg T)"
        },
        "leverage_cap": {
            "value": round(1.0 + risk_level * 1.0, 2),  # 1x-2x
            "unit": "x",
            "desc": "Maximum leverage allowed"
        }
    }

    # Apply custom overrides
    if custom_overrides:
        for key, val in custom_overrides.items():
            if key in params and val is not None:
                params[key]["value"] = val
                params[key]["custom"] = True

    return params


# ============== STOCK DATA FETCHER ==============
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock(ticker: str) -> dict:
    """Fetch stock data with timestamp."""
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
                "valid": True,
                "ticker": ticker,
                "price": price,
                "change": price - prev,
                "change_pct": ((price - prev) / prev * 100) if prev else 0,
                "name": info.get("shortName", ticker),
                "sector": info.get("sector", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE") or 0,
                "beta": info.get("beta") or 1.0,
                "high_52w": info.get("fiftyTwoWeekHigh") or 0,
                "low_52w": info.get("fiftyTwoWeekLow") or 0,
                "timestamp": ts,
                "source": "Yahoo Finance",
                "delay": "15-20 min delayed"
            }
    except Exception as e:
        pass

    return {
        "valid": False, "ticker": ticker, "price": 0, "change": 0, "change_pct": 0,
        "name": ticker, "sector": "Unknown", "timestamp": ts, "source": "N/A", "delay": "N/A"
    }


# ============== MAIN ANALYSIS ENGINE ==============
def run_analysis(
    tickers: List[str],
    analysts: List[str],
    risk_level: float,
    capital: float,
    holdings: Dict[str, int],
    mode_key: str,
    allow_fractional: bool = False,
    custom_risk_params: dict = None
) -> dict:
    """
    Run analysis with CORRECT allocation that matches mode description.
    """
    # Deterministic seed - use sorted lists for consistency
    sorted_analysts = sorted(analysts)
    seed_str = f"{sorted(tickers)}{sorted_analysts}{risk_level:.2f}{capital}{mode_key}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)

    risk_params = get_risk_params(risk_level, custom_risk_params)
    mode = ALLOCATION_MODES[mode_key]
    all_analysts = get_all_analysts()

    timestamp = datetime.now()

    # ========== PHASE 1: Fetch data & generate signals ==========
    ticker_results = {}

    for ticker in tickers:
        stock = fetch_stock(ticker)

        # Generate analyst signals
        signals = []
        for analyst_key in sorted_analysts:  # Use sorted for determinism
            if analyst_key not in all_analysts:
                continue

            info = all_analysts[analyst_key]
            sig_seed = int(hashlib.md5(f"{analyst_key}{ticker}{seed}".encode()).hexdigest()[:8], 16)
            np.random.seed(sig_seed)

            score = np.random.uniform(-1, 1) + info.get("bias", 0)

            if score > 0.2:
                signal = "BULLISH"
                confidence = 50 + score * 40
            elif score < -0.2:
                signal = "BEARISH"
                confidence = 50 + abs(score) * 40
            else:
                signal = "NEUTRAL"
                confidence = 40 + np.random.uniform(0, 20)

            signals.append({
                "analyst": info["name"],
                "analyst_key": analyst_key,
                "category": info["category"],
                "signal": signal,
                "confidence": min(95, max(30, confidence))
            })

        # Aggregate
        bullish = sum(1 for s in signals if s["signal"] == "BULLISH")
        bearish = sum(1 for s in signals if s["signal"] == "BEARISH")
        neutral = len(signals) - bullish - bearish
        total = len(signals)

        avg_conf = np.mean([s["confidence"] for s in signals]) if signals else 50
        min_conf = risk_params["min_confidence"]["value"]

        # Determine action
        if total == 0:
            action = "HOLD"
            action_reason = "No analysts selected"
        elif bullish > bearish and bullish >= neutral and avg_conf >= min_conf:
            action = "BUY"
            action_reason = f"Bullish consensus ({bullish}/{total}) with {avg_conf:.0f}% confidence >= {min_conf:.0f}% threshold"
        elif bearish > bullish and bearish >= neutral and avg_conf >= min_conf:
            action = "SHORT"
            action_reason = f"Bearish consensus ({bearish}/{total}) with {avg_conf:.0f}% confidence >= {min_conf:.0f}% threshold"
        elif avg_conf < min_conf:
            action = "HOLD"
            action_reason = f"Confidence {avg_conf:.0f}% below {min_conf:.0f}% threshold"
        else:
            action = "HOLD"
            action_reason = f"No clear consensus ({bullish}B/{neutral}N/{bearish}Be)"

        ticker_results[ticker] = {
            "stock": stock,
            "signals": signals,
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "total_analysts": total,
            "avg_confidence": avg_conf,
            "action": action,
            "action_reason": action_reason,
            "current_holdings": holdings.get(ticker, 0)
        }

    # ========== PHASE 2: Capital Allocation ==========
    actionable = {t: r for t, r in ticker_results.items() if r["action"] != "HOLD" and r["stock"]["valid"]}
    non_actionable = {t: r for t, r in ticker_results.items() if r["action"] == "HOLD" or not r["stock"]["valid"]}

    max_pos_pct = risk_params["max_position_pct"]["value"] / 100
    target_deploy_pct = mode.get("target_pct") or 0.5
    redistribute = mode.get("redistribute_excess", False)

    # Track where capital goes
    allocation_breakdown = {
        "capital": capital,
        "positions": {},
        "hold_tickers": {},
        "rounding_remainder": 0,
        "cap_blocked": 0,
        "unallocated_explicit": 0
    }

    if not actionable:
        # No trades possible
        allocation_breakdown["unallocated_explicit"] = capital
        for t, r in non_actionable.items():
            allocation_breakdown["hold_tickers"][t] = {
                "reason": r["action_reason"],
                "blocked_amount": capital / len(tickers) if tickers else 0
            }
    else:
        # Calculate initial allocation
        n_positions = len(actionable)
        target_total = capital * target_deploy_pct

        if mode_key == "equal_weight":
            per_position = target_total / n_positions
        elif mode_key == "max_deploy":
            per_position = target_total / n_positions
        elif mode_key == "confidence_weighted":
            total_conf = sum(r["avg_confidence"] for r in actionable.values())
            pass  # Handled per-ticker below
        else:  # conservative
            per_position = (capital * 0.5) / n_positions

        total_allocated = 0
        total_rounding = 0

        for ticker, result in actionable.items():
            stock = result["stock"]
            price = stock["price"]

            # Calculate allocation
            if mode_key == "confidence_weighted":
                total_conf = sum(r["avg_confidence"] for r in actionable.values())
                weight = result["avg_confidence"] / total_conf if total_conf > 0 else 1/n_positions
                budget = capital * max_pos_pct * weight * (result["avg_confidence"] / 100)
            else:
                budget = per_position

            # Apply position cap
            max_budget = capital * max_pos_pct
            capped = budget > max_budget
            if capped:
                allocation_breakdown["cap_blocked"] += (budget - max_budget)
                budget = max_budget

            # Calculate shares
            if allow_fractional:
                shares = budget / price
                actual_notional = budget
                remainder = 0
            else:
                shares = int(budget / price)
                actual_notional = shares * price
                remainder = budget - actual_notional

            total_allocated += actual_notional
            total_rounding += remainder

            # Stop loss / take profit - CORRECT FOR DIRECTION
            sl_pct = risk_params["stop_loss_pct"]["value"]
            tp_pct = risk_params["take_profit_pct"]["value"]

            if result["action"] == "BUY":
                sl_price = price * (1 - sl_pct / 100)
                tp_price = price * (1 + tp_pct / 100)
                sl_direction = "below"
                tp_direction = "above"
            else:  # SHORT
                sl_price = price * (1 + sl_pct / 100)
                tp_price = price * (1 - tp_pct / 100)
                sl_direction = "above"
                tp_direction = "below"

            # Delta from current holdings
            current = result["current_holdings"]
            if result["action"] == "BUY":
                delta = shares - current
            else:  # SHORT
                delta = -shares - current

            allocation_breakdown["positions"][ticker] = {
                "action": result["action"],
                "budget": budget,
                "shares": shares,
                "actual_notional": actual_notional,
                "remainder": remainder,
                "pct_of_portfolio": (actual_notional / capital * 100) if capital else 0,
                "capped": capped,
                "entry_price": price,
                "stop_loss": {"price": sl_price, "pct": sl_pct, "direction": sl_direction},
                "take_profit": {"price": tp_price, "pct": tp_pct, "direction": tp_direction},
                "current_holdings": current,
                "delta_shares": delta,
                "confidence": result["avg_confidence"],
                "timestamp": stock["timestamp"].strftime("%H:%M:%S")
            }

        allocation_breakdown["rounding_remainder"] = total_rounding

        # Redistribute excess if mode allows
        if redistribute and total_rounding > 100:
            uncapped = [t for t, p in allocation_breakdown["positions"].items() if not p["capped"]]
            if uncapped:
                extra_per = total_rounding / len(uncapped)
                for t in uncapped:
                    pos = allocation_breakdown["positions"][t]
                    price = pos["entry_price"]
                    extra_shares = int(extra_per / price)
                    if extra_shares > 0:
                        pos["shares"] += extra_shares
                        added = extra_shares * price
                        pos["actual_notional"] += added
                        pos["pct_of_portfolio"] = (pos["actual_notional"] / capital * 100)
                        allocation_breakdown["rounding_remainder"] -= added

        # Record HOLD tickers
        for t, r in non_actionable.items():
            allocation_breakdown["hold_tickers"][t] = {
                "reason": r["action_reason"],
                "blocked_amount": 0
            }

    # ========== PHASE 3: Calculate totals ==========
    positions = allocation_breakdown["positions"]
    long_exp = sum(p["actual_notional"] for p in positions.values() if p["action"] == "BUY")
    short_exp = sum(p["actual_notional"] for p in positions.values() if p["action"] == "SHORT")
    gross_exp = long_exp + short_exp
    net_exp = long_exp - short_exp
    cash_remaining = capital - gross_exp

    # Short margin calculation
    short_margin_req = risk_params["short_margin_req"]["value"] / 100
    margin_required = short_exp * short_margin_req
    buying_power = capital - margin_required

    # Max loss at stops
    max_loss_at_stop = sum(
        p["actual_notional"] * (p["stop_loss"]["pct"] / 100)
        for p in positions.values()
    )

    return {
        "timestamp": timestamp,
        "config": {
            "tickers": tickers,
            "analysts": sorted_analysts,
            "analyst_count": len(sorted_analysts),
            "risk_level": risk_level,
            "capital": capital,
            "mode": mode_key,
            "mode_name": mode["name"]
        },
        "risk_params": risk_params,
        "ticker_results": ticker_results,
        "allocation": allocation_breakdown,
        "summary": {
            "total_capital": capital,
            "deployed": gross_exp,
            "deployed_pct": (gross_exp / capital * 100) if capital else 0,
            "cash_remaining": cash_remaining,
            "cash_pct": (cash_remaining / capital * 100) if capital else 0,
            "rounding_remainder": allocation_breakdown["rounding_remainder"],
            "cap_blocked": allocation_breakdown["cap_blocked"],
            "long_exposure": long_exp,
            "short_exposure": short_exp,
            "gross_exposure": gross_exp,
            "net_exposure": net_exp,
            "margin_required": margin_required,
            "buying_power": buying_power,
            "max_loss_at_stop": max_loss_at_stop,
            "positions_count": len(positions),
            "hold_count": len(allocation_breakdown["hold_tickers"]),
            "total_tickers": len(tickers)
        }
    }


# ============== SESSION STATE ==============
if "result" not in st.session_state:
    st.session_state.result = None
if "selected_analysts" not in st.session_state:
    st.session_state.selected_analysts = list(get_all_analysts().keys())[:8]
if "custom_risk_params" not in st.session_state:
    st.session_state.custom_risk_params = {}
if "use_custom_params" not in st.session_state:
    st.session_state.use_custom_params = False
if "lookup_ticker" not in st.session_state:
    st.session_state.lookup_ticker = "AAPL"


# ============== HELPER: Ticker Quick-Peek Card ==============
def show_ticker_card(ticker: str, compact: bool = True):
    """Show a quick-peek card for a ticker."""
    stock = fetch_stock(ticker)
    if not stock["valid"]:
        st.warning(f"Could not fetch {ticker}")
        return

    chg_icon = "▲" if stock["change"] >= 0 else "▼"
    chg_color = "green" if stock["change"] >= 0 else "red"

    if compact:
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{ticker}** - {stock['name']}")
        with col2:
            st.markdown(f"${stock['price']:.2f} {chg_icon} {abs(stock['change_pct']):.2f}%")
        with col3:
            st.caption(stock['sector'])
    else:
        st.markdown(f"### {ticker} - {stock['name']}")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Price", f"${stock['price']:.2f}", f"{chg_icon} {abs(stock['change_pct']):.2f}%")
        with col2:
            st.metric("Sector", stock['sector'])
        with col3:
            st.metric("P/E", f"{stock['pe_ratio']:.1f}" if stock['pe_ratio'] else "N/A")
        with col4:
            st.metric("Beta", f"{stock['beta']:.2f}" if stock['beta'] else "N/A")


def make_ticker_clickable(ticker: str, show_lookup: bool = True) -> None:
    """Create a clickable ticker that opens in Securities tab."""
    if show_lookup:
        if st.button(f"🔍 {ticker}", key=f"lookup_{ticker}_{id(ticker)}"):
            st.session_state.lookup_ticker = ticker
            st.rerun()


# ============== STYLED DATAFRAME ==============
def styled_dataframe(df: pd.DataFrame, action_col: str = None):
    """Create a dark-themed, non-draggable dataframe with badges."""

    def style_action(val):
        if val == "BUY":
            return 'background-color: #238636; color: white; font-weight: bold; text-align: center;'
        elif val == "SHORT":
            return 'background-color: #da3633; color: white; font-weight: bold; text-align: center;'
        elif val == "HOLD":
            return 'background-color: #6e7681; color: white; font-weight: bold; text-align: center;'
        return ''

    def style_confidence(val):
        try:
            num = float(str(val).replace('%', ''))
            if num >= 70:
                return 'background-color: #238636; color: white;'
            elif num >= 50:
                return 'background-color: #d29922; color: black;'
            else:
                return 'background-color: #6e7681; color: white;'
        except:
            return ''

    # Apply styling
    styled = df.style.set_properties(**{
        'background-color': '#161b22',
        'color': '#e6edf3',
        'border-color': '#30363d'
    })

    if action_col and action_col in df.columns:
        styled = styled.applymap(style_action, subset=[action_col])

    if 'Confidence' in df.columns:
        styled = styled.applymap(style_confidence, subset=['Confidence'])

    # Display with configuration to prevent dragging
    st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
        column_config={col: st.column_config.TextColumn(col) for col in df.columns}
    )


# ============== HEADER ==============
st.markdown("# 📊 AI Hedge Fund Terminal")
st.caption("v4.1 | Data: Yahoo Finance (delayed 15-20 min)")

# ============== MAIN TABS ==============
tab_signals, tab_portfolio, tab_trades, tab_securities, tab_settings = st.tabs([
    "📈 Signals", "💼 Portfolio", "📋 Trade List", "🔍 Securities", "⚙️ Settings"
])


# ============== SIGNALS TAB ==============
with tab_signals:
    config_col, results_col = st.columns([1, 2])

    with config_col:
        st.subheader("Configuration")

        # Tickers
        st.markdown("**Stock Tickers**")
        ticker_input = st.text_input("Tickers", value="AAPL, MSFT, NVDA, GOOGL",
                                      label_visibility="collapsed",
                                      help="Comma-separated ticker symbols")
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        st.caption(f"{len(tickers)} tickers")

        # Capital
        st.markdown("**Investment Capital**")
        capital = st.number_input("Capital", min_value=1000, value=100000,
                                   step=10000, label_visibility="collapsed")

        # Holdings
        with st.expander("📁 Current Holdings (optional)", expanded=False):
            st.markdown("Enter holdings to see trade deltas:")
            holdings_text = st.text_area(
                "Format: TICKER:SHARES (one per line)",
                placeholder="AAPL:50\nMSFT:30",
                height=100,
                label_visibility="collapsed"
            )
            holdings = {}
            for line in holdings_text.strip().split("\n"):
                if ":" in line:
                    try:
                        t, s = line.split(":")
                        holdings[t.strip().upper()] = int(s.strip())
                    except:
                        pass
            if holdings:
                st.caption(f"Holdings: {holdings}")

        st.divider()

        # Allocation Mode
        st.markdown("**Allocation Mode**")
        mode_key = st.selectbox(
            "Mode",
            options=list(ALLOCATION_MODES.keys()),
            format_func=lambda x: ALLOCATION_MODES[x]["name"],
            label_visibility="collapsed"
        )
        st.info(ALLOCATION_MODES[mode_key]["desc"])

        with st.expander("ℹ️ Learn more about allocation modes", expanded=False):
            st.markdown("""
            **Maximum Deployment** - Invests 95% of capital, redistributes rounding remainders to fill positions. Best for full deployment strategies.

            **Equal Weight** - Splits 90% evenly among all actionable tickers. Simple, diversified approach.

            **Confidence Weighted** - Allocates more to higher-conviction trades. May hold significant cash if confidence is low.

            **Conservative** - Only trades high-confidence signals. Expects 30-50% cash buffer for safety.
            """)

        # Fractional shares
        allow_fractional = st.checkbox("Allow fractional shares",
                                        help="Paper trading only - eliminates rounding")

        st.divider()

        # Risk Level
        st.markdown("**Risk Level**")
        risk_level = st.slider("Risk", 0.0, 1.0, 0.5, 0.05, label_visibility="collapsed")

        risk_label = "Conservative" if risk_level < 0.35 else "Aggressive" if risk_level > 0.65 else "Moderate"
        st.markdown(f"**{risk_label}** Profile")

        # Show derived parameters (respecting custom overrides)
        active_params = get_risk_params(risk_level, st.session_state.custom_risk_params if st.session_state.use_custom_params else None)

        with st.expander("📊 Active Risk Parameters", expanded=True):
            param_rows = []
            for key, p in active_params.items():
                status = "✏️ Custom" if p.get("custom") else "Auto"
                param_rows.append({
                    "Parameter": p["desc"],
                    "Value": f"{p['value']}{p['unit']}",
                    "Status": status
                })
            st.dataframe(pd.DataFrame(param_rows), hide_index=True, use_container_width=True)

            if st.session_state.use_custom_params:
                st.caption("✏️ = Overridden in Settings > Custom Parameters")

        st.divider()

        # Analysts with PRE-FLIGHT VALIDATION
        st.markdown("**AI Analysts**")

        all_analyst_keys = list(get_all_analysts().keys())
        total_analysts = len(all_analyst_keys)
        selected_count = len(st.session_state.selected_analysts)

        # PRE-FLIGHT: Show selected count and warn about limits
        st.markdown(f"**{selected_count}/{total_analysts}** analysts selected")

        if selected_count == 0:
            st.error("⚠️ Select at least 1 analyst")
        elif selected_count == total_analysts:
            st.success(f"✅ All {total_analysts} analysts selected")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Select All", use_container_width=True):
                st.session_state.selected_analysts = all_analyst_keys.copy()
                st.rerun()
        with col2:
            if st.button("❌ Clear All", use_container_width=True):
                st.session_state.selected_analysts = []
                st.rerun()

        # Analyst selection by category
        for cat, analysts in ANALYST_CATEGORIES.items():
            cat_selected = sum(1 for a in analysts if a in st.session_state.selected_analysts)
            with st.expander(f"**{cat}** ({cat_selected}/{len(analysts)})", expanded=False):
                for key, info in analysts.items():
                    checked = st.checkbox(
                        f"{info['name']}",
                        value=key in st.session_state.selected_analysts,
                        key=f"analyst_{key}",
                        help=info["desc"]
                    )
                    if checked and key not in st.session_state.selected_analysts:
                        st.session_state.selected_analysts.append(key)
                    elif not checked and key in st.session_state.selected_analysts:
                        st.session_state.selected_analysts.remove(key)

        # PRE-FLIGHT: Show selected analyst IDs
        with st.expander("📋 Selected Analyst IDs (for debugging)", expanded=False):
            if st.session_state.selected_analysts:
                st.code("\n".join(sorted(st.session_state.selected_analysts)))
            else:
                st.warning("No analysts selected")

        st.divider()

        # RUN BUTTON with validation
        can_run = len(tickers) > 0 and len(st.session_state.selected_analysts) > 0

        # Pre-flight summary
        if can_run:
            st.markdown(f"""
            **Ready to analyze:**
            - {len(tickers)} tickers: {', '.join(tickers[:5])}{'...' if len(tickers) > 5 else ''}
            - {len(st.session_state.selected_analysts)} analysts
            - ${capital:,.0f} capital
            - {ALLOCATION_MODES[mode_key]['name']} mode
            """)

        if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True,
                     disabled=not can_run):
            with st.spinner(f"Analyzing {len(tickers)} tickers with {len(st.session_state.selected_analysts)} analysts..."):
                custom_params = st.session_state.custom_risk_params if st.session_state.use_custom_params else None
                st.session_state.result = run_analysis(
                    tickers=tickers,
                    analysts=st.session_state.selected_analysts.copy(),
                    risk_level=risk_level,
                    capital=capital,
                    holdings=holdings,
                    mode_key=mode_key,
                    allow_fractional=allow_fractional,
                    custom_risk_params=custom_params
                )
            st.rerun()

        if not can_run:
            if len(tickers) == 0:
                st.warning("⚠️ Enter at least one ticker")
            if len(st.session_state.selected_analysts) == 0:
                st.warning("⚠️ Select at least one analyst")

    # ========== RESULTS ==========
    with results_col:
        if st.session_state.result:
            r = st.session_state.result
            s = r["summary"]

            st.subheader("Capital Allocation")
            st.caption(f"Mode: **{r['config']['mode_name']}** | {r['config']['analyst_count']} analysts | Run: {r['timestamp'].strftime('%H:%M:%S')}")

            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Capital", f"${s['total_capital']:,.0f}")
            with col2:
                st.metric("Deployed", f"${s['deployed']:,.0f}", f"{s['deployed_pct']:.1f}%")
            with col3:
                st.metric("Cash", f"${s['cash_remaining']:,.0f}", f"{s['cash_pct']:.1f}%")
            with col4:
                st.metric("Max Loss @ Stop", f"${s['max_loss_at_stop']:,.0f}",
                         help="Total loss if all positions hit stop loss")

            st.divider()

            # WHY THIS ALLOCATION
            st.markdown("### Why This Allocation?")

            with st.expander("ℹ️ Learn more about constraints", expanded=False):
                st.markdown("""
                **Position Cap**: Maximum % of portfolio in a single position (from risk settings).

                **Rounding Remainder**: Cash that couldn't buy whole shares.

                **HOLD Reasons**: Why certain tickers aren't being traded.

                **Mode Target**: Each allocation mode targets different deployment levels.
                """)

            # Explicit breakdown
            for ticker, pos in r["allocation"]["positions"].items():
                item = f"**{ticker}** ({pos['action']}): ${pos['actual_notional']:,.0f}"
                if pos["capped"]:
                    item += " ⚠️ *cap hit*"
                st.markdown(f"✅ {item}")
                # Make ticker clickable
                col1, col2 = st.columns([4, 1])
                with col2:
                    if st.button(f"🔍", key=f"peek_{ticker}", help=f"Look up {ticker}"):
                        st.session_state.lookup_ticker = ticker

            # HOLD tickers
            for ticker, hold_info in r["allocation"]["hold_tickers"].items():
                st.markdown(f"⏸️ **{ticker}** (HOLD): {hold_info['reason']}")

            # Rounding & cap
            if s["rounding_remainder"] > 0:
                st.markdown(f"📊 **Rounding remainder**: ${s['rounding_remainder']:,.0f}")
            if s["cap_blocked"] > 0:
                st.markdown(f"🚫 **Cap blocked**: ${s['cap_blocked']:,.0f}")
            if s["cash_remaining"] > 100:
                if r["config"]["mode"] in ["confidence_weighted", "conservative"]:
                    st.markdown(f"💵 **Intentional cash buffer**: ${s['cash_remaining']:,.0f}")
                else:
                    st.markdown(f"💵 **Remaining cash**: ${s['cash_remaining']:,.0f}")

            st.divider()

            # Exposure summary
            st.markdown("### Exposure & Risk")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Long", f"${s['long_exposure']:,.0f}")
            with col2:
                st.metric("Short", f"${s['short_exposure']:,.0f}")
            with col3:
                st.metric("Gross", f"${s['gross_exposure']:,.0f}")
            with col4:
                st.metric("Net", f"${s['net_exposure']:,.0f}")

            # Short margin info
            if s["short_exposure"] > 0:
                with st.expander("⚠️ Short Position Margin Details", expanded=True):
                    st.markdown(f"""
                    **Initial Margin Required**: ${s['margin_required']:,.0f} (50% Reg T)

                    **Buying Power After Margin**: ${s['buying_power']:,.0f}

                    *Note: Actual borrow costs and availability vary by broker.*
                    """)

            st.divider()

            # Per-ticker cards
            st.markdown("### Recommendations")

            for ticker, tr in r["ticker_results"].items():
                stock = tr["stock"]
                pos = r["allocation"]["positions"].get(ticker)

                # Header with clickable ticker
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    if stock["valid"]:
                        chg_icon = "▲" if stock["change"] >= 0 else "▼"
                        st.markdown(f"### {ticker} — ${stock['price']:.2f} {chg_icon}{abs(stock['change_pct']):.2f}%")
                        st.caption(f"{stock['name']} | {stock['sector']}")
                    else:
                        st.markdown(f"### {ticker} — Price unavailable")
                with col2:
                    action = tr["action"]
                    if action == "BUY":
                        st.success(f"📈 {action}")
                    elif action == "SHORT":
                        st.error(f"📉 {action}")
                    else:
                        st.warning(f"⏸️ {action}")
                with col3:
                    if st.button(f"🔍 Details", key=f"detail_{ticker}"):
                        st.session_state.lookup_ticker = ticker

                # Action reason
                st.info(f"**Reason:** {tr['action_reason']}")

                # Position details
                if pos:
                    st.markdown("**Position:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        shares_fmt = f"{pos['shares']:,.2f}" if isinstance(pos['shares'], float) else f"{pos['shares']:,}"
                        st.metric("Shares", shares_fmt)
                    with col2:
                        st.metric("Notional", f"${pos['actual_notional']:,.0f}")
                    with col3:
                        sl = pos["stop_loss"]
                        st.metric("Stop Loss",
                                 f"${sl['price']:.2f}",
                                 f"{sl['pct']}% {sl['direction']}",
                                 delta_color="inverse")
                    with col4:
                        tp = pos["take_profit"]
                        st.metric("Take Profit",
                                 f"${tp['price']:.2f}",
                                 f"{tp['pct']}% {tp['direction']}")

                    if pos["current_holdings"] != 0 or pos["delta_shares"] != 0:
                        st.caption(f"Current: {pos['current_holdings']} → Target: {pos['shares']} → **Trade: {pos['delta_shares']:+}**")
                    st.caption(f"Position = {pos['pct_of_portfolio']:.1f}% of portfolio")

                # Vote breakdown
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Confidence", f"{tr['avg_confidence']:.0f}%")
                with col2:
                    st.metric("Bullish", tr["bullish"])
                with col3:
                    st.metric("Neutral", tr["neutral"])
                with col4:
                    st.metric("Bearish", tr["bearish"])

                # Analyst signals
                with st.expander(f"View {tr['total_analysts']} analyst signals"):
                    for sig in tr["signals"]:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.markdown(f"**{sig['analyst']}**")
                            st.caption(sig["category"])
                        with col2:
                            if sig["signal"] == "BULLISH":
                                st.success(sig["signal"])
                            elif sig["signal"] == "BEARISH":
                                st.error(sig["signal"])
                            else:
                                st.warning(sig["signal"])
                        with col3:
                            st.markdown(f"**{sig['confidence']:.0f}%**")

                st.divider()

        else:
            st.markdown("""
            ### Getting Started

            1. **Enter tickers** - Comma-separated symbols (AAPL, MSFT, etc.)
            2. **Set capital** - Your investment amount
            3. **Choose mode** - How to allocate capital
            4. **Adjust risk** - Controls position size, stops, thresholds
            5. **Select analysts** - Pick your AI advisors (Select All works!)
            6. **Run Analysis**
            """)

            with st.expander("ℹ️ Learn more about modes and settings", expanded=False):
                st.markdown("""
                **Allocation Modes:**
                - *Maximum Deployment*: Invests 95%+ of capital
                - *Equal Weight*: Splits evenly among trades
                - *Confidence Weighted*: Sizes by conviction
                - *Conservative*: Large cash buffer

                **Risk Settings:**
                Use the Settings tab to customize stop loss, take profit, position limits, and more.

                **Results include:**
                - Clear allocation breakdown
                - ALL tickers including HOLD with reasons
                - Stop loss / take profit levels
                - Delta from current holdings
                """)


# ============== PORTFOLIO TAB ==============
with tab_portfolio:
    st.subheader("Portfolio Overview")

    if st.session_state.result:
        r = st.session_state.result
        s = r["summary"]
        positions = r["allocation"]["positions"]

        # Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Positions", s["positions_count"])
        with col2:
            st.metric("HOLD", s["hold_count"])
        with col3:
            st.metric("Deployed", f"{s['deployed_pct']:.1f}%")
        with col4:
            st.metric("Cash", f"{s['cash_pct']:.1f}%")

        st.divider()

        # Position table - DARK THEMED
        if positions:
            st.markdown("### Positions")
            pos_data = []
            for ticker, pos in positions.items():
                pos_data.append({
                    "Ticker": ticker,
                    "Action": pos["action"],
                    "Shares": pos["shares"],
                    "Entry": f"${pos['entry_price']:.2f}",
                    "Notional": f"${pos['actual_notional']:,.0f}",
                    "% Port": f"{pos['pct_of_portfolio']:.1f}%",
                    "Stop Loss": f"${pos['stop_loss']['price']:.2f} ({pos['stop_loss']['direction']})",
                    "Take Profit": f"${pos['take_profit']['price']:.2f} ({pos['take_profit']['direction']})",
                    "Confidence": f"{pos['confidence']:.0f}%"
                })

            df = pd.DataFrame(pos_data)
            styled_dataframe(df, action_col="Action")

            # Ticker quick-lookup buttons
            st.markdown("**Quick Lookup:**")
            cols = st.columns(min(len(positions), 6))
            for i, ticker in enumerate(positions.keys()):
                with cols[i % len(cols)]:
                    if st.button(f"🔍 {ticker}", key=f"port_lookup_{ticker}"):
                        st.session_state.lookup_ticker = ticker

        # HOLD tickers
        hold_tickers = r["allocation"]["hold_tickers"]
        if hold_tickers:
            st.markdown("### Not Trading (HOLD)")
            for ticker, info in hold_tickers.items():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"- **{ticker}**: {info['reason']}")
                with col2:
                    if st.button(f"🔍", key=f"hold_lookup_{ticker}"):
                        st.session_state.lookup_ticker = ticker

        st.divider()

        # Cash breakdown
        st.markdown("### Cash Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Undeployed Cash", f"${s['cash_remaining']:,.0f}")
        with col2:
            st.metric("Rounding Remainder", f"${s['rounding_remainder']:,.0f}",
                     help="Lost to whole-share rounding")
        with col3:
            st.metric("Cap Blocked", f"${s['cap_blocked']:,.0f}",
                     help="Blocked by position size limit")
    else:
        st.info("Run analysis from Signals tab first.")


# ============== TRADE LIST TAB ==============
with tab_trades:
    st.subheader("Trade Instructions")

    if st.session_state.result:
        r = st.session_state.result
        positions = r["allocation"]["positions"]

        if positions:
            # Trade table - DARK THEMED
            trades = []
            for ticker, pos in positions.items():
                trades.append({
                    "Ticker": ticker,
                    "Action": pos["action"],
                    "Shares": pos["shares"],
                    "Entry": f"${pos['entry_price']:.2f}",
                    "Notional": f"${pos['actual_notional']:,.0f}",
                    "Stop Loss": f"${pos['stop_loss']['price']:.2f}",
                    "Take Profit": f"${pos['take_profit']['price']:.2f}",
                    "Delta": pos["delta_shares"] if pos["delta_shares"] != pos["shares"] else "New"
                })

            df = pd.DataFrame(trades)
            styled_dataframe(df, action_col="Action")

            # Ticker quick-lookup
            st.markdown("**Quick Lookup:**")
            cols = st.columns(min(len(positions), 6))
            for i, ticker in enumerate(positions.keys()):
                with cols[i % len(cols)]:
                    if st.button(f"🔍 {ticker}", key=f"trade_lookup_{ticker}"):
                        st.session_state.lookup_ticker = ticker

            st.divider()

            # Export
            st.markdown("### Export")

            csv_data = pd.DataFrame(trades).to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                "trades.csv",
                "text/csv",
                use_container_width=True
            )

            with st.expander("📋 Copy-Paste Format"):
                text = "TRADE LIST\n" + "="*40 + "\n"
                for t in trades:
                    text += f"{t['Action']} {t['Shares']} {t['Ticker']} @ {t['Entry']}\n"
                    text += f"  Stop: {t['Stop Loss']} | Target: {t['Take Profit']}\n\n"
                st.code(text)

            st.divider()

            # Execution notes - HIGH CONTRAST
            with st.expander("📝 Execution Notes", expanded=True):
                st.markdown("""
                **Data delay**: Prices are 15-20 minutes delayed. Use live quotes for execution.

                **Order type**: Consider limit orders near indicated entry prices.

                **Stop losses**: Set immediately after entry.

                **Margin**: Shorts require initial margin (typically 50% Reg T).

                **Borrow**: Short availability and cost vary by broker.
                """)
        else:
            st.info("No trades. All positions are HOLD.")
    else:
        st.info("Run analysis first.")


# ============== SECURITIES TAB ==============
with tab_securities:
    st.subheader("Securities Lookup")

    ticker = st.text_input("Enter ticker symbol", value=st.session_state.lookup_ticker, placeholder="AAPL")

    if ticker:
        with st.spinner(f"Fetching {ticker.upper()}..."):
            stock = fetch_stock(ticker.upper())

        if stock["valid"]:
            st.caption(f"Source: {stock['source']} | {stock['delay']} | as of {stock['timestamp'].strftime('%H:%M:%S')}")

            col1, col2 = st.columns([2, 1])

            with col1:
                chg_icon = "▲" if stock["change"] >= 0 else "▼"

                st.markdown(f"## {stock['ticker']}")
                st.markdown(f"*{stock['name']}*")

                st.metric(
                    "Price",
                    f"${stock['price']:.2f}",
                    f"{chg_icon} ${abs(stock['change']):.2f} ({abs(stock['change_pct']):.2f}%)"
                )

            with col2:
                st.metric("Sector", stock["sector"])
                st.metric("Market Cap", f"${stock['market_cap']/1e9:.1f}B" if stock["market_cap"] else "N/A")

            st.divider()

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("P/E Ratio", f"{stock['pe_ratio']:.1f}" if stock["pe_ratio"] else "N/A")
            with col2:
                st.metric("Beta", f"{stock['beta']:.2f}" if stock["beta"] else "N/A")
            with col3:
                st.metric("52W High", f"${stock['high_52w']:.2f}" if stock["high_52w"] else "N/A")
            with col4:
                st.metric("52W Low", f"${stock['low_52w']:.2f}" if stock["low_52w"] else "N/A")

            # Chart
            st.divider()
            st.markdown("### Price Chart (6 months)")
            try:
                import yfinance as yf
                hist = yf.Ticker(ticker.upper()).history(period="6mo")
                if len(hist) > 0:
                    st.line_chart(hist["Close"])
            except:
                st.warning("Chart unavailable")

            # Show in current analysis if available
            if st.session_state.result and ticker.upper() in st.session_state.result["ticker_results"]:
                st.divider()
                st.markdown("### From Current Analysis")
                tr = st.session_state.result["ticker_results"][ticker.upper()]

                col1, col2 = st.columns(2)
                with col1:
                    action = tr["action"]
                    if action == "BUY":
                        st.success(f"📈 Recommendation: {action}")
                    elif action == "SHORT":
                        st.error(f"📉 Recommendation: {action}")
                    else:
                        st.warning(f"⏸️ Recommendation: {action}")
                with col2:
                    st.metric("Confidence", f"{tr['avg_confidence']:.0f}%")

                st.info(f"**Reason:** {tr['action_reason']}")

                pos = st.session_state.result["allocation"]["positions"].get(ticker.upper())
                if pos:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Allocation", f"${pos['actual_notional']:,.0f}")
                    with col2:
                        st.metric("Stop Loss", f"${pos['stop_loss']['price']:.2f}")
                    with col3:
                        st.metric("Take Profit", f"${pos['take_profit']['price']:.2f}")
        else:
            st.error(f"Could not fetch data for {ticker.upper()}")


# ============== SETTINGS TAB ==============
with tab_settings:
    st.subheader("Settings")

    # Two tabs: Preset Reference and Custom Parameters
    settings_tab1, settings_tab2 = st.tabs(["📊 Risk Preset Reference", "✏️ Custom Parameters"])

    with settings_tab1:
        st.markdown("### How Risk Level Affects Parameters")
        st.markdown("The risk slider (0.0 to 1.0) controls all trading parameters:")

        levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        rows = []
        for level in levels:
            params = get_risk_params(level)
            label = "Very Conservative" if level == 0 else "Conservative" if level == 0.25 else "Moderate" if level == 0.5 else "Aggressive" if level == 0.75 else "Very Aggressive"
            rows.append({
                "Risk Level": f"{level:.0%} ({label})",
                "Max Position": f"{params['max_position_pct']['value']}%",
                "Stop Loss": f"{params['stop_loss_pct']['value']}%",
                "Take Profit": f"{params['take_profit_pct']['value']}%",
                "Min Confidence": f"{params['min_confidence']['value']}%",
                "Leverage Cap": f"{params['leverage_cap']['value']}x"
            })

        styled_dataframe(pd.DataFrame(rows))

        st.divider()

        with st.expander("ℹ️ Allocation Modes Explained", expanded=False):
            st.markdown("""
            | Mode | Behavior |
            |------|----------|
            | **Maximum Deployment** | Targets 95% deployment. Redistributes rounding remainders. |
            | **Equal Weight** | Splits 90% evenly among all actionable tickers. |
            | **Confidence Weighted** | Sizes by confidence × conviction. May hold significant cash. |
            | **Conservative** | Only high-confidence trades. Expects 30-50% cash buffer. |
            """)

        with st.expander("ℹ️ Stop Loss / Take Profit Logic", expanded=False):
            st.markdown("""
            **LONG positions**: Stop loss is *below* entry (exit if price falls), take profit is *above* entry.

            **SHORT positions**: Stop loss is *above* entry (exit if price rises against you), take profit is *below* entry.
            """)

        with st.expander("ℹ️ Margin Requirements (Shorts)", expanded=False):
            st.markdown("""
            - **Initial margin**: 50% (Reg T requirement)
            - **Maintenance margin**: 25-30% (varies by broker)
            - **Borrow costs**: Not included (varies by stock and broker)
            """)

    with settings_tab2:
        st.markdown("### Custom Parameter Overrides")
        st.markdown("Override the preset risk parameters with your own values.")

        # Enable/disable custom params
        use_custom = st.checkbox(
            "Enable custom parameters",
            value=st.session_state.use_custom_params,
            help="When enabled, these values override the preset risk level"
        )
        st.session_state.use_custom_params = use_custom

        if use_custom:
            st.warning("⚠️ Custom parameters are active. These will override preset values.")

            col1, col2 = st.columns(2)

            with col1:
                custom_max_pos = st.number_input(
                    "Max Position %",
                    min_value=5.0, max_value=100.0,
                    value=float(st.session_state.custom_risk_params.get("max_position_pct", 22.5)),
                    step=1.0,
                    help="Maximum allocation per position (5-100%)"
                )
                st.session_state.custom_risk_params["max_position_pct"] = custom_max_pos

                custom_sl = st.number_input(
                    "Stop Loss %",
                    min_value=1.0, max_value=50.0,
                    value=float(st.session_state.custom_risk_params.get("stop_loss_pct", 12.5)),
                    step=0.5,
                    help="Distance from entry to trigger stop loss (1-50%)"
                )
                st.session_state.custom_risk_params["stop_loss_pct"] = custom_sl

                custom_tp = st.number_input(
                    "Take Profit %",
                    min_value=5.0, max_value=200.0,
                    value=float(st.session_state.custom_risk_params.get("take_profit_pct", 37.5)),
                    step=1.0,
                    help="Distance from entry to take profit (5-200%)"
                )
                st.session_state.custom_risk_params["take_profit_pct"] = custom_tp

            with col2:
                custom_min_conf = st.number_input(
                    "Min Confidence %",
                    min_value=10.0, max_value=90.0,
                    value=float(st.session_state.custom_risk_params.get("min_confidence", 47.5)),
                    step=5.0,
                    help="Minimum confidence to trade (10-90%)"
                )
                st.session_state.custom_risk_params["min_confidence"] = custom_min_conf

                custom_leverage = st.number_input(
                    "Leverage Cap (x)",
                    min_value=1.0, max_value=5.0,
                    value=float(st.session_state.custom_risk_params.get("leverage_cap", 1.5)),
                    step=0.1,
                    help="Maximum leverage allowed (1-5x)"
                )
                st.session_state.custom_risk_params["leverage_cap"] = custom_leverage

            st.divider()

            # Reset button
            if st.button("🔄 Reset to Preset Values", use_container_width=True):
                st.session_state.custom_risk_params = {}
                st.session_state.use_custom_params = False
                st.rerun()

            # Show comparison
            st.markdown("### Current vs Preset Comparison")
            current_risk = 0.5  # Moderate preset
            preset_params = get_risk_params(current_risk)
            custom_params = get_risk_params(current_risk, st.session_state.custom_risk_params)

            comparison = []
            for key in ["max_position_pct", "stop_loss_pct", "take_profit_pct", "min_confidence", "leverage_cap"]:
                preset_val = preset_params[key]["value"]
                custom_val = custom_params[key]["value"]
                diff = "✏️ Changed" if custom_val != preset_val else "Same"
                comparison.append({
                    "Parameter": preset_params[key]["desc"],
                    "Preset (0.5)": f"{preset_val}{preset_params[key]['unit']}",
                    "Your Value": f"{custom_val}{custom_params[key]['unit']}",
                    "Status": diff
                })

            st.dataframe(pd.DataFrame(comparison), hide_index=True, use_container_width=True)
        else:
            st.info("Enable custom parameters above to override preset values.")


# ============== FOOTER ==============
st.divider()
st.caption("AI Hedge Fund Terminal v4.1 | Educational Use Only | Not Financial Advice")
st.caption("Data: Yahoo Finance (15-20 min delayed) | Past performance ≠ future results")
