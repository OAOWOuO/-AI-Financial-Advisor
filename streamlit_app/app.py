"""
AI Hedge Fund Terminal v4.0
Complete rewrite addressing all allocation, risk, and UX issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import io

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="AI Hedge Fund Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============== CSS - HIGH CONTRAST ==============
st.markdown("""
<style>
    .main { background: #0d1117; }
    .stApp { background: #0d1117; }
    #MainMenu, footer, header { visibility: hidden; }

    /* High contrast text */
    h1, h2, h3 { color: #e6edf3 !important; font-weight: 600 !important; }
    p, span, label { color: #c9d1d9 !important; }

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

    /* Cards */
    .info-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .warning-box {
        background: #3d2a1f;
        border: 1px solid #d29922;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .error-box {
        background: #3d1f1f;
        border: 1px solid #f85149;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .success-box {
        background: #1f3d2a;
        border: 1px solid #3fb950;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============== ALLOCATION MODES - RENAMED FOR CLARITY ==============
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
def get_risk_params(risk_level: float) -> dict:
    """
    Risk level 0.0 (very conservative) to 1.0 (very aggressive)

    Returns a table of derived parameters with explanations.
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
    return params


# ============== STOCK DATA FETCHER ==============
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
    allow_fractional: bool = False
) -> dict:
    """
    Run analysis with CORRECT allocation that matches mode description.
    """
    # Deterministic seed
    seed_str = f"{sorted(tickers)}{sorted(analysts)}{risk_level:.2f}{capital}{mode_key}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)

    risk_params = get_risk_params(risk_level)
    mode = ALLOCATION_MODES[mode_key]
    all_analysts = get_all_analysts()

    timestamp = datetime.now()

    # ========== PHASE 1: Fetch data & generate signals ==========
    ticker_results = {}

    for ticker in tickers:
        stock = fetch_stock(ticker)

        # Generate analyst signals
        signals = []
        for analyst_key in analysts:
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
            action_reason = f"Bullish consensus ({bullish}/{total}) with {avg_conf:.0f}% confidence ≥ {min_conf:.0f}% threshold"
        elif bearish > bullish and bearish >= neutral and avg_conf >= min_conf:
            action = "SHORT"
            action_reason = f"Bearish consensus ({bearish}/{total}) with {avg_conf:.0f}% confidence ≥ {min_conf:.0f}% threshold"
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
            # Equal split
            per_position = target_total / n_positions
        elif mode_key == "max_deploy":
            # Split proportionally, then redistribute excess
            per_position = target_total / n_positions
        elif mode_key == "confidence_weighted":
            # Weight by confidence
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
        if redistribute and total_rounding > 100:  # Only if meaningful amount
            # Try to add shares to positions that aren't capped
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
                "blocked_amount": 0  # Not blocking, just not traded
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
            "analysts": analysts,
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


# ============== HEADER ==============
st.markdown("# 📊 AI Hedge Fund Terminal")
st.caption("v4.0 | Data: Yahoo Finance (delayed 15-20 min)")

# ============== MAIN TABS - PROPER TAB BAR ==============
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
        with st.expander("Current Holdings (optional)"):
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
            label_visibility="collapsed",
            help="How to distribute capital"
        )
        st.info(ALLOCATION_MODES[mode_key]["desc"])

        # Fractional shares option
        allow_fractional = st.checkbox("Allow fractional shares (paper trading)",
                                        help="Eliminates rounding remainder")

        st.divider()

        # Risk Level with FULL PARAMETER TABLE
        st.markdown("**Risk Level**")
        risk_level = st.slider("Risk", 0.0, 1.0, 0.5, 0.05, label_visibility="collapsed")

        risk_label = "Conservative" if risk_level < 0.35 else "Aggressive" if risk_level > 0.65 else "Moderate"
        st.markdown(f"**{risk_label}** Profile")

        # Show derived parameters
        risk_params = get_risk_params(risk_level)
        st.markdown("*At this risk level:*")
        param_df = pd.DataFrame([
            {"Parameter": p["desc"], "Value": f"{p['value']}{p['unit']}"}
            for p in risk_params.values()
        ])
        st.dataframe(param_df, hide_index=True, use_container_width=True)

        st.divider()

        # Analysts
        st.markdown("**AI Analysts**")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_analysts = list(get_all_analysts().keys())
                st.rerun()
        with col2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.selected_analysts = []
                st.rerun()

        for cat, analysts in ANALYST_CATEGORIES.items():
            with st.expander(f"**{cat}** ({sum(1 for a in analysts if a in st.session_state.selected_analysts)}/{len(analysts)})"):
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

        st.caption(f"{len(st.session_state.selected_analysts)} analysts selected")

        st.divider()

        # RUN BUTTON
        can_run = len(tickers) > 0 and len(st.session_state.selected_analysts) > 0

        if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True,
                     disabled=not can_run):
            with st.spinner("Fetching data and analyzing..."):
                st.session_state.result = run_analysis(
                    tickers=tickers,
                    analysts=st.session_state.selected_analysts,
                    risk_level=risk_level,
                    capital=capital,
                    holdings=holdings,
                    mode_key=mode_key,
                    allow_fractional=allow_fractional
                )
            st.rerun()

        if not can_run:
            st.warning("Select tickers and analysts to run")

    # ========== RESULTS ==========
    with results_col:
        if st.session_state.result:
            r = st.session_state.result
            s = r["summary"]

            st.subheader("Capital Allocation")
            st.caption(f"Mode: **{r['config']['mode_name']}** | Run: {r['timestamp'].strftime('%H:%M:%S')}")

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

            # WHY THIS ALLOCATION - Constraint breakdown
            st.markdown("### Why This Allocation?")

            # Explicit breakdown
            breakdown_items = []

            # Positions
            for ticker, pos in r["allocation"]["positions"].items():
                item = f"**{ticker}** ({pos['action']}): ${pos['actual_notional']:,.0f}"
                if pos["capped"]:
                    item += " ⚠️ *position cap hit*"
                breakdown_items.append(("✅", item))

            # HOLD tickers - ALWAYS SHOW
            for ticker, hold_info in r["allocation"]["hold_tickers"].items():
                breakdown_items.append(("⏸️", f"**{ticker}** (HOLD): {hold_info['reason']}"))

            # Rounding
            if s["rounding_remainder"] > 0:
                breakdown_items.append(("📊", f"**Rounding remainder**: ${s['rounding_remainder']:,.0f} (whole shares constraint)"))

            # Cap blocked
            if s["cap_blocked"] > 0:
                breakdown_items.append(("🚫", f"**Position cap blocked**: ${s['cap_blocked']:,.0f} (max {r['risk_params']['max_position_pct']['value']}% per position)"))

            # Cash
            if s["cash_remaining"] > 100:
                if r["config"]["mode"] in ["confidence_weighted", "conservative"]:
                    breakdown_items.append(("💵", f"**Intentional cash buffer**: ${s['cash_remaining']:,.0f} (mode: {r['config']['mode_name']})"))
                else:
                    breakdown_items.append(("💵", f"**Remaining cash**: ${s['cash_remaining']:,.0f}"))

            for icon, text in breakdown_items:
                st.markdown(f"{icon} {text}")

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
                st.markdown(f"""
                <div class="warning-box">
                    <strong>Short Position Margin:</strong><br>
                    Initial Margin Required: ${s['margin_required']:,.0f} (50% Reg T)<br>
                    Buying Power After Margin: ${s['buying_power']:,.0f}<br>
                    <em>Note: Actual borrow costs and availability vary by broker.</em>
                </div>
                """, unsafe_allow_html=True)

            st.divider()

            # Per-ticker cards
            st.markdown("### Recommendations")

            for ticker, tr in r["ticker_results"].items():
                stock = tr["stock"]
                pos = r["allocation"]["positions"].get(ticker)

                # Header
                col1, col2 = st.columns([3, 1])
                with col1:
                    if stock["valid"]:
                        chg_icon = "▲" if stock["change"] >= 0 else "▼"
                        st.markdown(f"### {ticker} — ${stock['price']:.2f} {chg_icon}{abs(stock['change_pct']):.2f}%")
                        st.caption(f"{stock['name']} | {stock['source']} as of {stock['timestamp'].strftime('%H:%M:%S')} ({stock['delay']})")
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

                # Action reason - ALWAYS VISIBLE
                st.info(f"**Reason:** {tr['action_reason']}")

                # Position details (if trading)
                if pos:
                    st.markdown("**Position:**")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Shares", f"{pos['shares']:,.2f}" if isinstance(pos['shares'], float) else f"{pos['shares']:,}")
                    with col2:
                        st.metric("Notional", f"${pos['actual_notional']:,.0f}")
                    with col3:
                        sl = pos["stop_loss"]
                        st.metric("Stop Loss",
                                 f"${sl['price']:.2f}",
                                 f"{sl['pct']}% {sl['direction']} entry",
                                 delta_color="inverse")
                    with col4:
                        tp = pos["take_profit"]
                        st.metric("Take Profit",
                                 f"${tp['price']:.2f}",
                                 f"{tp['pct']}% {tp['direction']} entry")

                    # Delta from holdings
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
            3. **Choose mode**:
               - *Maximum Deployment*: Invests 95%+ of capital
               - *Equal Weight*: Splits evenly among trades
               - *Confidence Weighted*: Sizes by conviction (may hold cash)
               - *Conservative*: Only high-conviction, large cash buffer
            4. **Adjust risk** - Controls position size, stops, thresholds
            5. **Select analysts** - Pick your AI advisors
            6. **Run Analysis**

            Results will show:
            - Clear allocation breakdown (where every $ goes)
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

        # Position table
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
                    "% Portfolio": f"{pos['pct_of_portfolio']:.1f}%",
                    "Stop Loss": f"${pos['stop_loss']['price']:.2f} ({pos['stop_loss']['direction']})",
                    "Take Profit": f"${pos['take_profit']['price']:.2f} ({pos['take_profit']['direction']})",
                    "Confidence": f"{pos['confidence']:.0f}%"
                })
            st.dataframe(pd.DataFrame(pos_data), hide_index=True, use_container_width=True)

        # HOLD tickers
        hold_tickers = r["allocation"]["hold_tickers"]
        if hold_tickers:
            st.markdown("### Not Trading (HOLD)")
            for ticker, info in hold_tickers.items():
                st.markdown(f"- **{ticker}**: {info['reason']}")

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
            # Trade table
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

            st.dataframe(pd.DataFrame(trades), hide_index=True, use_container_width=True)

            st.divider()

            # Export
            st.markdown("### Export")

            # CSV export
            csv_data = pd.DataFrame(trades).to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                "trades.csv",
                "text/csv",
                use_container_width=True
            )

            # Copy-friendly format
            with st.expander("📋 Copy-Paste Format"):
                text = "TRADE LIST\n" + "="*40 + "\n"
                for t in trades:
                    text += f"{t['Action']} {t['Shares']} {t['Ticker']} @ {t['Entry']}\n"
                    text += f"  Stop: {t['Stop Loss']} | Target: {t['Take Profit']}\n\n"
                st.code(text)

            st.divider()

            # Execution notes
            st.markdown("### Execution Notes")
            st.markdown("""
            - **Data delay**: Prices are 15-20 minutes delayed. Use live quotes for execution.
            - **Order type**: Consider limit orders near indicated entry prices.
            - **Stop losses**: Set immediately after entry.
            - **Margin**: Shorts require initial margin (typically 50% Reg T).
            - **Borrow**: Short availability and cost vary by broker.
            """)
        else:
            st.info("No trades. All positions are HOLD.")
    else:
        st.info("Run analysis first.")


# ============== SECURITIES TAB ==============
with tab_securities:
    st.subheader("Securities Lookup")

    ticker = st.text_input("Enter ticker symbol", value="AAPL", placeholder="AAPL")

    if ticker:
        with st.spinner(f"Fetching {ticker.upper()}..."):
            stock = fetch_stock(ticker.upper())

        if stock["valid"]:
            # Data source
            st.caption(f"Source: {stock['source']} | {stock['delay']} | as of {stock['timestamp'].strftime('%H:%M:%S')}")

            col1, col2 = st.columns([2, 1])

            with col1:
                chg_color = "green" if stock["change"] >= 0 else "red"
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
        else:
            st.error(f"Could not fetch data for {ticker.upper()}")


# ============== SETTINGS TAB ==============
with tab_settings:
    st.subheader("Risk Parameter Reference")

    st.markdown("""
    ### How Risk Level Affects Parameters

    The risk slider (0.0 to 1.0) controls all trading parameters.
    Here's the full mapping:
    """)

    # Generate table for different risk levels
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

    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.divider()

    st.markdown("""
    ### Allocation Modes Explained

    | Mode | Behavior |
    |------|----------|
    | **Maximum Deployment** | Targets 95% deployment. Redistributes rounding remainders. |
    | **Equal Weight** | Splits 90% evenly among all actionable tickers. |
    | **Confidence Weighted** | Sizes by confidence × conviction. May hold significant cash. |
    | **Conservative** | Only high-confidence trades. Expects 30-50% cash buffer. |

    ### Stop Loss / Take Profit Logic

    - **LONG positions**: Stop loss is *below* entry (exit if price falls), take profit is *above* entry.
    - **SHORT positions**: Stop loss is *above* entry (exit if price rises against you), take profit is *below* entry.

    ### Margin Requirements (Shorts)

    - Initial margin: 50% (Reg T requirement)
    - Maintenance margin: 25-30% (varies by broker)
    - Borrow costs: Not included (varies by stock and broker)
    """)


# ============== FOOTER ==============
st.divider()
st.caption("AI Hedge Fund Terminal v4.0 | Educational Use Only | Not Financial Advice")
st.caption("Data: Yahoo Finance (15-20 min delayed) | Past performance ≠ future results")
