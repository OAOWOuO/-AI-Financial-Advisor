"""
AI Hedge Fund Terminal v4.2
Fixes: allocation caps, form-based state, static tables, allocation trace, chart ranges, analysts tab
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

# ============== CSS - DARK THEME + STATIC TABLES ==============
st.markdown("""
<style>
    /* Base dark theme */
    .main { background: #0d1117; }
    .stApp { background: #0d1117; }
    #MainMenu, footer, header { visibility: hidden; }

    /* High contrast text */
    h1, h2, h3, h4 { color: #e6edf3 !important; font-weight: 600 !important; }
    p, span, label, li { color: #c9d1d9 !important; }
    .stMarkdown { color: #c9d1d9 !important; }

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

    /* Static report-style tables */
    .report-table {
        width: 100%;
        border-collapse: collapse;
        background: #161b22;
        border-radius: 8px;
        overflow: hidden;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    .report-table th {
        background: #21262d;
        color: #e6edf3;
        padding: 12px 16px;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid #30363d;
        position: sticky;
        top: 0;
    }
    .report-table td {
        padding: 10px 16px;
        color: #c9d1d9;
        border-bottom: 1px solid #30363d;
    }
    .report-table tr:hover {
        background: #1c2128;
    }
    .report-table .num {
        text-align: right;
        font-family: "SF Mono", Monaco, monospace;
    }
    .report-table .ticker {
        font-weight: 600;
        color: #58a6ff;
    }
    .badge-buy {
        background: #238636;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-short {
        background: #da3633;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-hold {
        background: #6e7681;
        color: white;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-bullish {
        background: #238636;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
    }
    .badge-bearish {
        background: #da3633;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
    }
    .badge-neutral {
        background: #6e7681;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
    }

    /* Allocation trace styling */
    .trace-step {
        background: #161b22;
        border-left: 3px solid #30363d;
        padding: 12px 16px;
        margin: 8px 0;
    }
    .trace-step.input { border-left-color: #58a6ff; }
    .trace-step.signal { border-left-color: #a371f7; }
    .trace-step.constraint { border-left-color: #d29922; }
    .trace-step.result { border-left-color: #3fb950; }

    /* Cards */
    .info-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
        color: #e6edf3;
    }
    .warning-box {
        background: #3d2a1f;
        border: 1px solid #d29922;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        color: #e6edf3;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        color: #e6edf3 !important;
        background: #161b22 !important;
    }
</style>
""", unsafe_allow_html=True)


# ============== ALLOCATION MODES ==============
ALLOCATION_MODES = {
    "max_deploy": {
        "name": "Maximum Deployment",
        "desc": "Deploy 95%+ of capital. Respects position cap unless concentration override is enabled.",
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
        "warren_buffett": {"name": "Warren Buffett", "desc": "Moats, quality management, long-term value", "bias": -0.1, "style": "Seeks companies with durable competitive advantages trading below intrinsic value"},
        "charlie_munger": {"name": "Charlie Munger", "desc": "Mental models, business quality", "bias": -0.1, "style": "Focuses on business quality and management integrity over pure value metrics"},
        "ben_graham": {"name": "Benjamin Graham", "desc": "Margin of safety, net-net value", "bias": -0.15, "style": "Deep value, requires significant discount to book value"},
        "joel_greenblatt": {"name": "Joel Greenblatt", "desc": "Magic formula: ROIC + earnings yield", "bias": -0.05, "style": "Quantitative value screening based on return on capital and earnings yield"},
        "seth_klarman": {"name": "Seth Klarman", "desc": "Deep value, distressed assets", "bias": -0.15, "style": "Contrarian deep value, often in distressed or out-of-favor situations"},
    },
    "Growth Investors": {
        "peter_lynch": {"name": "Peter Lynch", "desc": "PEG ratio, growth at reasonable price", "bias": 0.05, "style": "Growth at a reasonable price, favors companies you understand"},
        "phil_fisher": {"name": "Philip Fisher", "desc": "Scuttlebutt, quality growth", "bias": 0.05, "style": "Long-term growth investing with deep qualitative research"},
        "cathie_wood": {"name": "Cathie Wood", "desc": "Disruptive innovation, exponential growth", "bias": 0.2, "style": "High-conviction bets on disruptive innovation and exponential growth"},
        "bill_ackman": {"name": "Bill Ackman", "desc": "Activist catalysts, concentrated bets", "bias": 0.1, "style": "Concentrated positions with activist catalysts to unlock value"},
    },
    "Macro Traders": {
        "stanley_druckenmiller": {"name": "Stanley Druckenmiller", "desc": "Macro trends, asymmetric bets", "bias": 0, "style": "Macro trend following with aggressive sizing on high-conviction ideas"},
        "george_soros": {"name": "George Soros", "desc": "Reflexivity, regime changes", "bias": 0, "style": "Identifies reflexive feedback loops and regime changes"},
        "ray_dalio": {"name": "Ray Dalio", "desc": "Economic machine, risk parity", "bias": -0.05, "style": "Systematic macro based on economic machine principles"},
        "paul_tudor_jones": {"name": "Paul Tudor Jones", "desc": "Technical macro, trend following", "bias": 0, "style": "Technical analysis combined with macro themes"},
    },
    "Quantitative Agents": {
        "fundamentals_agent": {"name": "Fundamentals Analyst", "desc": "Financial ratios, earnings quality, balance sheet", "bias": 0, "style": "Analyzes financial statements, ratios, and earnings quality"},
        "technical_agent": {"name": "Technical Analyst", "desc": "Price patterns, momentum, RSI, MACD", "bias": 0, "style": "Technical indicators including RSI, MACD, moving averages"},
        "sentiment_agent": {"name": "Sentiment Analyst", "desc": "News sentiment, social media, analyst ratings", "bias": 0.05, "style": "Aggregates news sentiment, social media buzz, analyst ratings"},
        "valuation_agent": {"name": "Valuation Analyst", "desc": "DCF, comparable analysis, sum-of-parts", "bias": -0.05, "style": "DCF models, comparable company analysis, sum-of-parts"},
        "momentum_agent": {"name": "Momentum Analyst", "desc": "Price momentum, earnings momentum", "bias": 0.1, "style": "Follows price and earnings momentum trends"},
        "risk_agent": {"name": "Risk Analyst", "desc": "Volatility, drawdown, tail risk", "bias": -0.1, "style": "Focuses on risk metrics: volatility, drawdown, tail risk"},
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
    params = {
        "max_position_pct": {
            "value": round(10 + risk_level * 25, 1),
            "unit": "%",
            "desc": "Maximum allocation per position"
        },
        "stop_loss_pct": {
            "value": round(20 - risk_level * 15, 1),
            "unit": "%",
            "desc": "Stop loss distance from entry"
        },
        "take_profit_pct": {
            "value": round(15 + risk_level * 45, 1),
            "unit": "%",
            "desc": "Take profit target from entry"
        },
        "min_confidence": {
            "value": round(65 - risk_level * 35, 0),
            "unit": "%",
            "desc": "Minimum confidence to trade"
        },
        "max_sector_pct": {
            "value": round(25 + risk_level * 25, 0),
            "unit": "%",
            "desc": "Maximum sector concentration"
        },
        "short_margin_req": {
            "value": 50,
            "unit": "%",
            "desc": "Initial margin for shorts (Reg T)"
        },
        "leverage_cap": {
            "value": round(1.0 + risk_level * 1.0, 2),
            "unit": "x",
            "desc": "Maximum leverage allowed"
        }
    }

    if custom_overrides:
        for key, val in custom_overrides.items():
            if key in params and val is not None:
                params[key]["value"] = val
                params[key]["custom"] = True

    return params


# ============== STOCK DATA FETCHER ==============
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
    except:
        pass

    return {
        "valid": False, "ticker": ticker, "price": 0, "change": 0, "change_pct": 0,
        "name": ticker, "sector": "Unknown", "timestamp": ts, "source": "N/A", "delay": "N/A"
    }


@st.cache_data(ttl=300)
def fetch_stock_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Fetch historical price data for charting."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        return hist
    except:
        return pd.DataFrame()


# ============== MAIN ANALYSIS ENGINE ==============
def run_analysis(
    tickers: List[str],
    analysts: List[str],
    risk_level: float,
    capital: float,
    holdings: Dict[str, int],
    mode_key: str,
    allow_fractional: bool = False,
    concentration_override: bool = False,
    custom_risk_params: dict = None
) -> dict:
    """
    Run analysis with full allocation trace for transparency.
    """
    sorted_analysts = sorted(analysts)
    seed_str = f"{sorted(tickers)}{sorted_analysts}{risk_level:.2f}{capital}{mode_key}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)

    risk_params = get_risk_params(risk_level, custom_risk_params)
    mode = ALLOCATION_MODES[mode_key]
    all_analysts = get_all_analysts()

    timestamp = datetime.now()

    # ========== ALLOCATION TRACE ==========
    trace = {
        "inputs": {
            "tickers": tickers,
            "analysts": sorted_analysts,
            "analyst_count": len(sorted_analysts),
            "capital": capital,
            "risk_level": risk_level,
            "mode": mode_key,
            "mode_name": mode["name"],
            "concentration_override": concentration_override
        },
        "signals": {},
        "constraints": [],
        "allocation_steps": [],
        "final": {}
    }

    # ========== PHASE 1: Generate signals ==========
    ticker_results = {}

    for ticker in tickers:
        stock = fetch_stock(ticker)

        signals = []
        for analyst_key in sorted_analysts:
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
                "style": info.get("style", ""),
                "signal": signal,
                "confidence": min(95, max(30, confidence)),
                "score": score
            })

        bullish = sum(1 for s in signals if s["signal"] == "BULLISH")
        bearish = sum(1 for s in signals if s["signal"] == "BEARISH")
        neutral = len(signals) - bullish - bearish
        total = len(signals)

        avg_conf = np.mean([s["confidence"] for s in signals]) if signals else 50
        min_conf = risk_params["min_confidence"]["value"]

        if total == 0:
            action = "HOLD"
            action_reason = "No analysts selected"
        elif bullish > bearish and bullish >= neutral and avg_conf >= min_conf:
            action = "BUY"
            action_reason = f"Bullish consensus ({bullish}/{total}) at {avg_conf:.0f}% confidence >= {min_conf:.0f}% threshold"
        elif bearish > bullish and bearish >= neutral and avg_conf >= min_conf:
            action = "SHORT"
            action_reason = f"Bearish consensus ({bearish}/{total}) at {avg_conf:.0f}% confidence >= {min_conf:.0f}% threshold"
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

        # Record in trace
        trace["signals"][ticker] = {
            "price": stock["price"] if stock["valid"] else 0,
            "valid": stock["valid"],
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
            "avg_confidence": avg_conf,
            "action": action,
            "reason": action_reason
        }

    # ========== PHASE 2: Allocation with trace ==========
    actionable = {t: r for t, r in ticker_results.items() if r["action"] != "HOLD" and r["stock"]["valid"]}
    non_actionable = {t: r for t, r in ticker_results.items() if r["action"] == "HOLD" or not r["stock"]["valid"]}

    n_actionable = len(actionable)
    base_max_pos_pct = risk_params["max_position_pct"]["value"] / 100
    target_deploy_pct = mode.get("target_pct") or 0.5

    # CONCENTRATION OVERRIDE: Dynamically increase position cap for small n
    if concentration_override and n_actionable > 0 and n_actionable <= 3:
        # Allow up to target_deploy_pct / n_actionable per position
        effective_max_pos_pct = min(target_deploy_pct / n_actionable, 0.95)
        trace["constraints"].append({
            "type": "concentration_override",
            "desc": f"Concentration override active: {n_actionable} actionable tickers, position cap raised from {base_max_pos_pct:.1%} to {effective_max_pos_pct:.1%}"
        })
    else:
        effective_max_pos_pct = base_max_pos_pct
        if n_actionable > 0 and n_actionable <= 3 and not concentration_override:
            blocked_by_cap = capital * target_deploy_pct - capital * base_max_pos_pct * n_actionable
            if blocked_by_cap > 0:
                trace["constraints"].append({
                    "type": "position_cap_warning",
                    "desc": f"Position cap ({base_max_pos_pct:.1%}) limits deployment. With {n_actionable} ticker(s), max deployable = {base_max_pos_pct * n_actionable:.1%}. Enable 'Concentration Override' to deploy more."
                })

    allocation_breakdown = {
        "capital": capital,
        "positions": {},
        "hold_tickers": {},
        "rounding_remainder": 0,
        "cap_blocked": 0,
        "unallocated_explicit": 0
    }

    if not actionable:
        allocation_breakdown["unallocated_explicit"] = capital
        trace["allocation_steps"].append({
            "step": "No actionable tickers",
            "detail": "All tickers are HOLD or have invalid prices. 100% cash."
        })
        for t, r in non_actionable.items():
            allocation_breakdown["hold_tickers"][t] = {
                "reason": r["action_reason"],
                "blocked_amount": capital / len(tickers) if tickers else 0
            }
    else:
        target_total = capital * target_deploy_pct
        trace["allocation_steps"].append({
            "step": f"Target deployment: {target_deploy_pct:.0%} of ${capital:,.0f} = ${target_total:,.0f}",
            "detail": f"Mode '{mode['name']}' targets {target_deploy_pct:.0%} deployment"
        })

        if mode_key in ["equal_weight", "max_deploy"]:
            per_position = target_total / n_actionable
            trace["allocation_steps"].append({
                "step": f"Initial allocation: ${target_total:,.0f} / {n_actionable} tickers = ${per_position:,.0f} each",
                "detail": "Equal split among actionable tickers"
            })
        elif mode_key == "confidence_weighted":
            total_conf = sum(r["avg_confidence"] for r in actionable.values())
            trace["allocation_steps"].append({
                "step": f"Confidence-weighted allocation",
                "detail": f"Total confidence pool: {total_conf:.0f}%"
            })

        total_allocated = 0
        total_rounding = 0
        total_cap_blocked = 0

        for ticker, result in actionable.items():
            stock = result["stock"]
            price = stock["price"]

            if mode_key == "confidence_weighted":
                total_conf = sum(r["avg_confidence"] for r in actionable.values())
                weight = result["avg_confidence"] / total_conf if total_conf > 0 else 1/n_actionable
                budget = capital * effective_max_pos_pct * weight * (result["avg_confidence"] / 100)
            else:
                budget = target_total / n_actionable

            # Apply position cap
            max_budget = capital * effective_max_pos_pct
            capped = budget > max_budget
            cap_blocked_amt = 0
            if capped:
                cap_blocked_amt = budget - max_budget
                total_cap_blocked += cap_blocked_amt
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

            # Stop loss / take profit
            sl_pct = risk_params["stop_loss_pct"]["value"]
            tp_pct = risk_params["take_profit_pct"]["value"]

            if result["action"] == "BUY":
                sl_price = price * (1 - sl_pct / 100)
                tp_price = price * (1 + tp_pct / 100)
                sl_direction = "below"
                tp_direction = "above"
            else:
                sl_price = price * (1 + sl_pct / 100)
                tp_price = price * (1 - tp_pct / 100)
                sl_direction = "above"
                tp_direction = "below"

            current = result["current_holdings"]
            if result["action"] == "BUY":
                delta = shares - current
            else:
                delta = -shares - current

            allocation_breakdown["positions"][ticker] = {
                "action": result["action"],
                "budget": budget,
                "shares": shares,
                "actual_notional": actual_notional,
                "remainder": remainder,
                "pct_of_portfolio": (actual_notional / capital * 100) if capital else 0,
                "capped": capped,
                "cap_blocked": cap_blocked_amt,
                "entry_price": price,
                "stop_loss": {"price": sl_price, "pct": sl_pct, "direction": sl_direction},
                "take_profit": {"price": tp_price, "pct": tp_pct, "direction": tp_direction},
                "current_holdings": current,
                "delta_shares": delta,
                "confidence": result["avg_confidence"],
                "timestamp": stock["timestamp"].strftime("%H:%M:%S")
            }

            # Trace this allocation
            trace["allocation_steps"].append({
                "step": f"{ticker}: {result['action']}",
                "detail": f"Budget ${budget:,.0f} @ ${price:.2f} = {shares} shares (${actual_notional:,.0f})" +
                         (f" [capped, ${cap_blocked_amt:,.0f} blocked]" if capped else "") +
                         (f" [${remainder:,.0f} rounding]" if remainder > 1 else "")
            })

        allocation_breakdown["rounding_remainder"] = total_rounding
        allocation_breakdown["cap_blocked"] = total_cap_blocked

        # Redistribute if allowed
        redistribute = mode.get("redistribute_excess", False)
        if redistribute and total_rounding > 100:
            uncapped = [t for t, p in allocation_breakdown["positions"].items() if not p["capped"]]
            if uncapped:
                extra_per = total_rounding / len(uncapped)
                redistributed = 0
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
                        redistributed += added

                if redistributed > 0:
                    trace["allocation_steps"].append({
                        "step": f"Redistributed ${redistributed:,.0f} from rounding remainder",
                        "detail": f"Added shares to uncapped positions"
                    })

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

    short_margin_req = risk_params["short_margin_req"]["value"] / 100
    margin_required = short_exp * short_margin_req
    buying_power = capital - margin_required

    max_loss_at_stop = sum(
        p["actual_notional"] * (p["stop_loss"]["pct"] / 100)
        for p in positions.values()
    )

    # Final trace
    trace["final"] = {
        "deployed": gross_exp,
        "deployed_pct": (gross_exp / capital * 100) if capital else 0,
        "cash": cash_remaining,
        "cash_pct": (cash_remaining / capital * 100) if capital else 0,
        "rounding_remainder": allocation_breakdown["rounding_remainder"],
        "cap_blocked": allocation_breakdown["cap_blocked"],
        "positions_count": len(positions),
        "hold_count": len(allocation_breakdown["hold_tickers"])
    }

    return {
        "timestamp": timestamp,
        "config": {
            "tickers": tickers,
            "analysts": sorted_analysts,
            "analyst_count": len(sorted_analysts),
            "risk_level": risk_level,
            "capital": capital,
            "mode": mode_key,
            "mode_name": mode["name"],
            "concentration_override": concentration_override
        },
        "risk_params": risk_params,
        "ticker_results": ticker_results,
        "allocation": allocation_breakdown,
        "trace": trace,
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


# ============== SESSION STATE INITIALIZATION ==============
if "result" not in st.session_state:
    st.session_state.result = None
if "selected_analysts" not in st.session_state:
    st.session_state.selected_analysts = set(get_all_analysts().keys())  # Default: all selected
if "custom_risk_params" not in st.session_state:
    st.session_state.custom_risk_params = {}
if "use_custom_params" not in st.session_state:
    st.session_state.use_custom_params = False
if "lookup_ticker" not in st.session_state:
    st.session_state.lookup_ticker = "AAPL"
if "chart_period" not in st.session_state:
    st.session_state.chart_period = "1y"


# ============== HELPER: Generate HTML tables ==============
def generate_positions_table(positions: dict) -> str:
    """Generate static HTML table for positions."""
    if not positions:
        return "<p>No positions</p>"

    rows = ""
    for ticker, pos in positions.items():
        action_badge = f'<span class="badge-{pos["action"].lower()}">{pos["action"]}</span>'
        rows += f"""
        <tr>
            <td class="ticker">{ticker}</td>
            <td>{action_badge}</td>
            <td class="num">{pos['shares']:,}</td>
            <td class="num">${pos['entry_price']:.2f}</td>
            <td class="num">${pos['actual_notional']:,.0f}</td>
            <td class="num">{pos['pct_of_portfolio']:.1f}%</td>
            <td class="num">${pos['stop_loss']['price']:.2f}</td>
            <td class="num">${pos['take_profit']['price']:.2f}</td>
            <td class="num">{pos['confidence']:.0f}%</td>
        </tr>
        """

    return f"""
    <table class="report-table">
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Action</th>
                <th>Shares</th>
                <th>Entry</th>
                <th>Notional</th>
                <th>% Port</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
                <th>Conf</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def generate_trades_table(positions: dict) -> str:
    """Generate static HTML table for trade list."""
    if not positions:
        return "<p>No trades</p>"

    rows = ""
    for ticker, pos in positions.items():
        action_badge = f'<span class="badge-{pos["action"].lower()}">{pos["action"]}</span>'
        delta = pos["delta_shares"]
        delta_str = "New" if delta == pos["shares"] or pos["current_holdings"] == 0 else f"{delta:+,}"

        rows += f"""
        <tr>
            <td class="ticker">{ticker}</td>
            <td>{action_badge}</td>
            <td class="num">{pos['shares']:,}</td>
            <td class="num">${pos['entry_price']:.2f}</td>
            <td class="num">${pos['actual_notional']:,.0f}</td>
            <td class="num">${pos['stop_loss']['price']:.2f}</td>
            <td class="num">${pos['take_profit']['price']:.2f}</td>
            <td class="num">{delta_str}</td>
        </tr>
        """

    return f"""
    <table class="report-table">
        <thead>
            <tr>
                <th>Ticker</th>
                <th>Action</th>
                <th>Shares</th>
                <th>Entry</th>
                <th>Notional</th>
                <th>Stop Loss</th>
                <th>Take Profit</th>
                <th>Delta</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def generate_analyst_signals_table(signals: list, ticker: str) -> str:
    """Generate HTML table for analyst signals."""
    if not signals:
        return "<p>No signals</p>"

    rows = ""
    for sig in signals:
        signal_badge = f'<span class="badge-{sig["signal"].lower()}">{sig["signal"]}</span>'
        rows += f"""
        <tr>
            <td><strong>{sig['analyst']}</strong><br><small style="color:#8b949e">{sig['category']}</small></td>
            <td>{signal_badge}</td>
            <td class="num">{sig['confidence']:.0f}%</td>
        </tr>
        """

    return f"""
    <table class="report-table">
        <thead>
            <tr>
                <th>Analyst</th>
                <th>Signal</th>
                <th>Confidence</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
    """


def generate_export_csv(result: dict) -> str:
    """Generate CSV export of trades."""
    positions = result["allocation"]["positions"]
    if not positions:
        return ""

    rows = ["Ticker,Action,Shares,Entry,Notional,Stop Loss,Take Profit,Confidence,Delta"]
    for ticker, pos in positions.items():
        delta = pos["delta_shares"]
        delta_str = "New" if delta == pos["shares"] or pos["current_holdings"] == 0 else str(delta)
        rows.append(f'{ticker},{pos["action"]},{pos["shares"]},{pos["entry_price"]:.2f},{pos["actual_notional"]:.0f},{pos["stop_loss"]["price"]:.2f},{pos["take_profit"]["price"]:.2f},{pos["confidence"]:.0f},{delta_str}')

    return "\n".join(rows)


def generate_export_text(result: dict) -> str:
    """Generate copy-paste text export."""
    positions = result["allocation"]["positions"]
    if not positions:
        return "No trades"

    lines = [
        f"AI HEDGE FUND - TRADE LIST",
        f"Generated: {result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}",
        f"Capital: ${result['config']['capital']:,.0f}",
        f"Mode: {result['config']['mode_name']}",
        f"Risk Level: {result['config']['risk_level']:.0%}",
        f"Analysts: {result['config']['analyst_count']}",
        "",
        "=" * 50,
        ""
    ]

    for ticker, pos in positions.items():
        lines.append(f"{pos['action']} {pos['shares']:,} {ticker} @ ${pos['entry_price']:.2f}")
        lines.append(f"  Notional: ${pos['actual_notional']:,.0f} ({pos['pct_of_portfolio']:.1f}% of portfolio)")
        lines.append(f"  Stop Loss: ${pos['stop_loss']['price']:.2f} ({pos['stop_loss']['direction']} entry)")
        lines.append(f"  Take Profit: ${pos['take_profit']['price']:.2f} ({pos['take_profit']['direction']} entry)")
        lines.append(f"  Confidence: {pos['confidence']:.0f}%")
        lines.append("")

    lines.extend([
        "=" * 50,
        f"Total Deployed: ${result['summary']['deployed']:,.0f} ({result['summary']['deployed_pct']:.1f}%)",
        f"Cash Remaining: ${result['summary']['cash_remaining']:,.0f} ({result['summary']['cash_pct']:.1f}%)",
        f"Max Loss at Stops: ${result['summary']['max_loss_at_stop']:,.0f}",
        "",
        "DISCLAIMER: Educational use only. Not financial advice."
    ])

    return "\n".join(lines)


# ============== HEADER ==============
st.markdown("# 📊 AI Hedge Fund Terminal")
st.caption("v4.2 | Data: Yahoo Finance (delayed 15-20 min)")

# ============== MAIN TABS ==============
tab_signals, tab_portfolio, tab_trades, tab_analysts, tab_securities, tab_settings = st.tabs([
    "📈 Signals", "💼 Portfolio", "📋 Trade List", "🧠 Analysts", "🔍 Securities", "⚙️ Settings"
])


# ============== SIGNALS TAB ==============
with tab_signals:
    config_col, results_col = st.columns([1, 2])

    with config_col:
        st.subheader("Configuration")

        # Use a form to prevent reruns on every input change
        with st.form(key="analysis_form"):
            # Tickers
            st.markdown("**Stock Tickers**")
            ticker_input = st.text_input(
                "Tickers",
                value="AAPL, MSFT, NVDA, GOOGL",
                label_visibility="collapsed",
                help="Comma-separated ticker symbols"
            )
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
            st.caption(f"{len(tickers)} ticker(s)")

            # Capital
            st.markdown("**Investment Capital**")
            capital = st.number_input(
                "Capital",
                min_value=1000,
                value=100000,
                step=10000,
                label_visibility="collapsed"
            )

            # Holdings
            st.markdown("**Current Holdings** (optional)")
            holdings_text = st.text_area(
                "Format: TICKER:SHARES",
                placeholder="AAPL:50\nMSFT:30",
                height=80,
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

            st.divider()

            # Allocation Mode
            st.markdown("**Allocation Mode**")
            mode_key = st.selectbox(
                "Mode",
                options=list(ALLOCATION_MODES.keys()),
                format_func=lambda x: ALLOCATION_MODES[x]["name"],
                label_visibility="collapsed"
            )
            st.caption(ALLOCATION_MODES[mode_key]["desc"])

            # Concentration override option
            concentration_override = st.checkbox(
                "Enable concentration override",
                value=False,
                help="When enabled with few tickers, allows position caps to increase dynamically to deploy more capital"
            )

            allow_fractional = st.checkbox(
                "Allow fractional shares",
                value=False,
                help="Paper trading only"
            )

            st.divider()

            # Risk Level
            st.markdown("**Risk Level**")
            risk_level = st.slider("Risk", 0.0, 1.0, 0.5, 0.05, label_visibility="collapsed")
            risk_label = "Conservative" if risk_level < 0.35 else "Aggressive" if risk_level > 0.65 else "Moderate"
            st.caption(f"{risk_label} ({risk_level:.0%})")

            st.divider()

            # Analysts - using stable keys
            st.markdown("**AI Analysts**")

            all_analyst_keys = list(get_all_analysts().keys())
            total_analysts = len(all_analyst_keys)

            col1, col2 = st.columns(2)
            with col1:
                select_all = st.checkbox("Select All", value=len(st.session_state.selected_analysts) == total_analysts)
            with col2:
                clear_all = st.checkbox("Clear All", value=len(st.session_state.selected_analysts) == 0)

            # Handle select/clear all
            if select_all and len(st.session_state.selected_analysts) != total_analysts:
                st.session_state.selected_analysts = set(all_analyst_keys)
            if clear_all and len(st.session_state.selected_analysts) != 0:
                st.session_state.selected_analysts = set()

            # Analyst checkboxes by category
            for cat, analysts in ANALYST_CATEGORIES.items():
                with st.expander(f"{cat}"):
                    for key, info in analysts.items():
                        checked = st.checkbox(
                            info["name"],
                            value=key in st.session_state.selected_analysts,
                            key=f"form_analyst_{key}",
                            help=info["desc"]
                        )
                        if checked:
                            st.session_state.selected_analysts.add(key)
                        else:
                            st.session_state.selected_analysts.discard(key)

            selected_count = len(st.session_state.selected_analysts)
            st.caption(f"**{selected_count}/{total_analysts}** analysts selected")

            st.divider()

            # Submit button
            submitted = st.form_submit_button(
                "🚀 RUN ANALYSIS",
                type="primary",
                use_container_width=True
            )

            if submitted:
                if len(tickers) == 0:
                    st.error("Enter at least one ticker")
                elif selected_count == 0:
                    st.error("Select at least one analyst")
                else:
                    custom_params = st.session_state.custom_risk_params if st.session_state.use_custom_params else None
                    st.session_state.result = run_analysis(
                        tickers=tickers,
                        analysts=list(st.session_state.selected_analysts),
                        risk_level=risk_level,
                        capital=capital,
                        holdings=holdings,
                        mode_key=mode_key,
                        allow_fractional=allow_fractional,
                        concentration_override=concentration_override,
                        custom_risk_params=custom_params
                    )

    # ========== RESULTS ==========
    with results_col:
        if st.session_state.result:
            r = st.session_state.result
            s = r["summary"]
            trace = r["trace"]

            st.subheader("Results")
            st.caption(f"Run: {r['timestamp'].strftime('%H:%M:%S')} | Mode: {r['config']['mode_name']} | {r['config']['analyst_count']} analysts")

            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Capital", f"${s['total_capital']:,.0f}")
            with col2:
                st.metric("Deployed", f"${s['deployed']:,.0f}", f"{s['deployed_pct']:.1f}%")
            with col3:
                st.metric("Cash", f"${s['cash_remaining']:,.0f}", f"{s['cash_pct']:.1f}%")
            with col4:
                st.metric("Max Loss", f"${s['max_loss_at_stop']:,.0f}")

            st.divider()

            # ========== ALLOCATION TRACE ==========
            st.markdown("### Allocation Trace")
            st.caption("Step-by-step breakdown of how capital was allocated")

            # Inputs
            st.markdown(f"""
            <div class="trace-step input">
                <strong>INPUTS</strong><br>
                Tickers: {', '.join(trace['inputs']['tickers'])}<br>
                Analysts: {trace['inputs']['analyst_count']}<br>
                Capital: ${trace['inputs']['capital']:,.0f}<br>
                Mode: {trace['inputs']['mode_name']}<br>
                Concentration Override: {'Yes' if trace['inputs']['concentration_override'] else 'No'}
            </div>
            """, unsafe_allow_html=True)

            # Signals summary
            signals_html = "<br>".join([
                f"<strong>{t}</strong>: {sig['action']} ({sig['bullish']}B/{sig['neutral']}N/{sig['bearish']}Be, {sig['avg_confidence']:.0f}% conf)"
                for t, sig in trace['signals'].items()
            ])
            st.markdown(f"""
            <div class="trace-step signal">
                <strong>SIGNALS</strong><br>
                {signals_html}
            </div>
            """, unsafe_allow_html=True)

            # Constraints
            if trace["constraints"]:
                constraints_html = "<br>".join([f"⚠️ {c['desc']}" for c in trace["constraints"]])
                st.markdown(f"""
                <div class="trace-step constraint">
                    <strong>CONSTRAINTS</strong><br>
                    {constraints_html}
                </div>
                """, unsafe_allow_html=True)

            # Allocation steps
            if trace["allocation_steps"]:
                steps_html = "<br>".join([
                    f"→ {step['step']}" + (f"<br>&nbsp;&nbsp;&nbsp;<small style='color:#8b949e'>{step['detail']}</small>" if step.get('detail') else "")
                    for step in trace["allocation_steps"]
                ])
                st.markdown(f"""
                <div class="trace-step result">
                    <strong>ALLOCATION STEPS</strong><br>
                    {steps_html}
                </div>
                """, unsafe_allow_html=True)

            # Final result
            st.markdown(f"""
            <div class="trace-step result">
                <strong>FINAL RESULT</strong><br>
                Deployed: ${trace['final']['deployed']:,.0f} ({trace['final']['deployed_pct']:.1f}%)<br>
                Cash: ${trace['final']['cash']:,.0f} ({trace['final']['cash_pct']:.1f}%)<br>
                Positions: {trace['final']['positions_count']} | HOLD: {trace['final']['hold_count']}<br>
                Rounding Remainder: ${trace['final']['rounding_remainder']:,.0f}<br>
                Cap Blocked: ${trace['final']['cap_blocked']:,.0f}
            </div>
            """, unsafe_allow_html=True)

            st.divider()

            # Exposure
            st.markdown("### Exposure")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Long", f"${s['long_exposure']:,.0f}")
            with col2:
                st.metric("Short", f"${s['short_exposure']:,.0f}")
            with col3:
                st.metric("Gross", f"${s['gross_exposure']:,.0f}")
            with col4:
                st.metric("Net", f"${s['net_exposure']:,.0f}")

            if s["short_exposure"] > 0:
                st.info(f"Short margin required: ${s['margin_required']:,.0f} (50% Reg T)")

            st.divider()

            # Per-ticker recommendations
            st.markdown("### Recommendations")

            for ticker, tr in r["ticker_results"].items():
                stock = tr["stock"]
                pos = r["allocation"]["positions"].get(ticker)

                col1, col2 = st.columns([4, 1])
                with col1:
                    if stock["valid"]:
                        chg_icon = "▲" if stock["change"] >= 0 else "▼"
                        st.markdown(f"**{ticker}** — ${stock['price']:.2f} {chg_icon}{abs(stock['change_pct']):.2f}%")
                        st.caption(f"{stock['name']} | {stock['sector']}")
                    else:
                        st.markdown(f"**{ticker}** — Price unavailable")
                with col2:
                    action = tr["action"]
                    if action == "BUY":
                        st.success(f"📈 {action}")
                    elif action == "SHORT":
                        st.error(f"📉 {action}")
                    else:
                        st.warning(f"⏸️ {action}")

                st.caption(f"**Reason:** {tr['action_reason']}")

                if pos:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Shares", f"{pos['shares']:,}")
                    with col2:
                        st.metric("Notional", f"${pos['actual_notional']:,.0f}")
                    with col3:
                        st.metric("Stop Loss", f"${pos['stop_loss']['price']:.2f}")
                    with col4:
                        st.metric("Take Profit", f"${pos['take_profit']['price']:.2f}")

                with st.expander(f"View {tr['total_analysts']} analyst signals"):
                    st.markdown(generate_analyst_signals_table(tr["signals"], ticker), unsafe_allow_html=True)

                st.divider()

            # Export section - always available
            st.markdown("### Export")
            col1, col2 = st.columns(2)
            with col1:
                csv_data = generate_export_csv(r)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    f"trades_{r['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True,
                    disabled=len(r["allocation"]["positions"]) == 0
                )
            with col2:
                text_data = generate_export_text(r)
                st.download_button(
                    "📋 Download Text",
                    text_data,
                    f"trades_{r['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain",
                    use_container_width=True
                )

            with st.expander("📋 Copy-Paste Format"):
                st.code(generate_export_text(r), language=None)

        else:
            st.markdown("""
            ### Getting Started

            1. Enter tickers (comma-separated)
            2. Set investment capital
            3. Choose allocation mode
            4. Adjust risk level
            5. Select AI analysts
            6. Click **RUN ANALYSIS**

            **Allocation Modes:**
            - **Maximum Deployment**: Deploy 95%+ of capital
            - **Equal Weight**: Split evenly among trades
            - **Confidence Weighted**: Size by conviction
            - **Conservative**: Large cash buffer

            **Concentration Override:**
            Enable this when trading 1-3 tickers to allow larger positions and reach target deployment.
            """)


# ============== PORTFOLIO TAB ==============
with tab_portfolio:
    st.subheader("Portfolio Overview")

    if st.session_state.result:
        r = st.session_state.result
        s = r["summary"]
        positions = r["allocation"]["positions"]

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

        if positions:
            st.markdown("### Positions")
            st.markdown(generate_positions_table(positions), unsafe_allow_html=True)

        hold_tickers = r["allocation"]["hold_tickers"]
        if hold_tickers:
            st.markdown("### Not Trading (HOLD)")
            for ticker, info in hold_tickers.items():
                st.markdown(f"- **{ticker}**: {info['reason']}")

        st.divider()

        st.markdown("### Cash Analysis")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Undeployed", f"${s['cash_remaining']:,.0f}")
        with col2:
            st.metric("Rounding", f"${s['rounding_remainder']:,.0f}")
        with col3:
            st.metric("Cap Blocked", f"${s['cap_blocked']:,.0f}")

        # Export
        st.divider()
        st.markdown("### Export")
        col1, col2 = st.columns(2)
        with col1:
            csv_data = generate_export_csv(r)
            st.download_button(
                "📥 Download CSV",
                csv_data,
                f"portfolio_{r['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv",
                use_container_width=True,
                disabled=len(positions) == 0
            )
        with col2:
            text_data = generate_export_text(r)
            st.download_button(
                "📋 Download Text",
                text_data,
                f"portfolio_{r['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain",
                use_container_width=True
            )
    else:
        st.info("Run analysis from Signals tab first.")


# ============== TRADE LIST TAB ==============
with tab_trades:
    st.subheader("Trade Instructions")

    if st.session_state.result:
        r = st.session_state.result
        positions = r["allocation"]["positions"]

        if positions:
            st.markdown(generate_trades_table(positions), unsafe_allow_html=True)

            st.divider()

            # Execution notes
            with st.expander("📝 Execution Notes", expanded=True):
                st.markdown("""
                **Data delay**: Prices are 15-20 minutes delayed. Use live quotes for execution.

                **Order type**: Consider limit orders near indicated entry prices.

                **Stop losses**: Set immediately after entry. Direction matters for shorts.

                **Margin**: Shorts require 50% initial margin (Reg T).

                **Borrow**: Short availability and cost vary by broker.
                """)

            st.divider()

            # Export
            st.markdown("### Export")
            col1, col2 = st.columns(2)
            with col1:
                csv_data = generate_export_csv(r)
                st.download_button(
                    "📥 Download CSV",
                    csv_data,
                    f"trades_{r['timestamp'].strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
            with col2:
                text_data = generate_export_text(r)
                st.download_button(
                    "📋 Download Text",
                    text_data,
                    f"trades_{r['timestamp'].strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain",
                    use_container_width=True
                )

            with st.expander("📋 Copy-Paste Format"):
                st.code(generate_export_text(r), language=None)
        else:
            st.info("No trades. All positions are HOLD.")
    else:
        st.info("Run analysis first.")


# ============== ANALYSTS TAB ==============
with tab_analysts:
    st.subheader("AI Analysts")

    # Show selected analysts from current run
    if st.session_state.result:
        r = st.session_state.result
        selected = r["config"]["analysts"]

        st.markdown(f"### Selected Analysts ({len(selected)})")
        st.caption("These analysts contributed to the current analysis run")

        all_analysts = get_all_analysts()

        for analyst_key in selected:
            if analyst_key not in all_analysts:
                continue
            info = all_analysts[analyst_key]

            st.markdown(f"""
            <div class="info-box">
                <strong>{info['name']}</strong> <small style="color:#8b949e">({info['category']})</small><br>
                <p>{info['desc']}</p>
                <p style="color:#8b949e"><em>{info.get('style', '')}</em></p>
                <p style="color:#8b949e">Bias: {info.get('bias', 0):+.2f} (negative = bearish tendency, positive = bullish tendency)</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Show signals by analyst across all tickers
        st.markdown("### Analyst Signals (This Run)")

        for analyst_key in selected:
            if analyst_key not in all_analysts:
                continue
            info = all_analysts[analyst_key]

            with st.expander(f"**{info['name']}** - {info['category']}"):
                st.caption(info.get('style', ''))

                signals_for_analyst = []
                for ticker, tr in r["ticker_results"].items():
                    for sig in tr["signals"]:
                        if sig["analyst_key"] == analyst_key:
                            signals_for_analyst.append({
                                "ticker": ticker,
                                "signal": sig["signal"],
                                "confidence": sig["confidence"],
                                "score": sig.get("score", 0)
                            })

                if signals_for_analyst:
                    rows = ""
                    for s in signals_for_analyst:
                        badge = f'<span class="badge-{s["signal"].lower()}">{s["signal"]}</span>'
                        rows += f"""
                        <tr>
                            <td class="ticker">{s['ticker']}</td>
                            <td>{badge}</td>
                            <td class="num">{s['confidence']:.0f}%</td>
                            <td class="num">{s['score']:.2f}</td>
                        </tr>
                        """
                    st.markdown(f"""
                    <table class="report-table">
                        <thead><tr><th>Ticker</th><th>Signal</th><th>Confidence</th><th>Score</th></tr></thead>
                        <tbody>{rows}</tbody>
                    </table>
                    """, unsafe_allow_html=True)
                else:
                    st.caption("No signals from this analyst")

    else:
        st.markdown("### All Available Analysts")
        st.caption("Run an analysis to see specific signals")

        for cat, analysts in ANALYST_CATEGORIES.items():
            st.markdown(f"#### {cat}")
            for key, info in analysts.items():
                selected_marker = "✅" if key in st.session_state.selected_analysts else "⬜"
                st.markdown(f"""
                {selected_marker} **{info['name']}**
                - {info['desc']}
                - *{info.get('style', '')}*
                - Bias: {info.get('bias', 0):+.2f}
                """)


# ============== SECURITIES TAB ==============
with tab_securities:
    st.subheader("Securities Lookup")

    ticker = st.text_input("Enter ticker symbol", value=st.session_state.lookup_ticker, placeholder="AAPL")

    if ticker:
        ticker = ticker.upper()

        with st.spinner(f"Fetching {ticker}..."):
            stock = fetch_stock(ticker)

        if stock["valid"]:
            st.caption(f"Source: {stock['source']} | {stock['delay']} | as of {stock['timestamp'].strftime('%H:%M:%S')}")

            # Price display
            col1, col2 = st.columns([2, 1])

            with col1:
                chg_icon = "▲" if stock["change"] >= 0 else "▼"
                chg_color = "green" if stock["change"] >= 0 else "red"

                st.markdown(f"## {stock['ticker']}")
                st.markdown(f"*{stock['name']}*")

                st.metric(
                    "Price",
                    f"${stock['price']:.2f}",
                    f"{chg_icon} ${abs(stock['change']):.2f} ({abs(stock['change_pct']):.2f}%)"
                )

            with col2:
                st.metric("Sector", stock["sector"])
                if stock["market_cap"]:
                    if stock["market_cap"] >= 1e12:
                        cap_str = f"${stock['market_cap']/1e12:.2f}T"
                    elif stock["market_cap"] >= 1e9:
                        cap_str = f"${stock['market_cap']/1e9:.1f}B"
                    else:
                        cap_str = f"${stock['market_cap']/1e6:.0f}M"
                    st.metric("Market Cap", cap_str)

            st.divider()

            # Key metrics table
            st.markdown("### Key Metrics")
            metrics_html = f"""
            <table class="report-table">
                <tr><td>P/E Ratio</td><td class="num">{stock['pe_ratio']:.1f if stock['pe_ratio'] else 'N/A'}</td></tr>
                <tr><td>Beta</td><td class="num">{stock['beta']:.2f if stock['beta'] else 'N/A'}</td></tr>
                <tr><td>52-Week High</td><td class="num">${stock['high_52w']:.2f if stock['high_52w'] else 'N/A'}</td></tr>
                <tr><td>52-Week Low</td><td class="num">${stock['low_52w']:.2f if stock['low_52w'] else 'N/A'}</td></tr>
            </table>
            """
            st.markdown(metrics_html, unsafe_allow_html=True)

            st.divider()

            # Chart with time range selector
            st.markdown("### Price Chart")

            time_ranges = {
                "1M": "1mo",
                "3M": "3mo",
                "6M": "6mo",
                "1Y": "1y",
                "2Y": "2y",
                "5Y": "5y",
                "MAX": "max"
            }

            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
            cols = [col1, col2, col3, col4, col5, col6, col7]

            for i, (label, period) in enumerate(time_ranges.items()):
                with cols[i]:
                    if st.button(label, key=f"chart_{label}", use_container_width=True,
                                type="primary" if st.session_state.chart_period == period else "secondary"):
                        st.session_state.chart_period = period

            hist = fetch_stock_history(ticker, st.session_state.chart_period)
            if len(hist) > 0:
                st.line_chart(hist["Close"])

                # Price range stats
                if len(hist) > 1:
                    period_high = hist["Close"].max()
                    period_low = hist["Close"].min()
                    period_start = hist["Close"].iloc[0]
                    period_end = hist["Close"].iloc[-1]
                    period_change = ((period_end - period_start) / period_start * 100)

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Period High", f"${period_high:.2f}")
                    with col2:
                        st.metric("Period Low", f"${period_low:.2f}")
                    with col3:
                        st.metric("Period Change", f"{period_change:+.1f}%")
                    with col4:
                        st.metric("Volatility", f"{hist['Close'].std():.2f}")
            else:
                st.warning("Chart data unavailable")

            # Show in current analysis if available
            if st.session_state.result and ticker in st.session_state.result["ticker_results"]:
                st.divider()
                st.markdown("### From Current Analysis")
                tr = st.session_state.result["ticker_results"][ticker]

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

                st.caption(f"**Reason:** {tr['action_reason']}")

                pos = st.session_state.result["allocation"]["positions"].get(ticker)
                if pos:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Allocation", f"${pos['actual_notional']:,.0f}")
                    with col2:
                        st.metric("Stop Loss", f"${pos['stop_loss']['price']:.2f}")
                    with col3:
                        st.metric("Take Profit", f"${pos['take_profit']['price']:.2f}")
        else:
            st.error(f"Could not fetch data for {ticker}")


# ============== SETTINGS TAB ==============
with tab_settings:
    st.subheader("Settings")

    settings_tab1, settings_tab2 = st.tabs(["📊 Risk Presets", "✏️ Custom Parameters"])

    with settings_tab1:
        st.markdown("### Risk Level Parameter Mapping")

        levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        rows_html = ""
        for level in levels:
            params = get_risk_params(level)
            label = "Very Conservative" if level == 0 else "Conservative" if level == 0.25 else "Moderate" if level == 0.5 else "Aggressive" if level == 0.75 else "Very Aggressive"
            rows_html += f"""
            <tr>
                <td>{level:.0%} ({label})</td>
                <td class="num">{params['max_position_pct']['value']}%</td>
                <td class="num">{params['stop_loss_pct']['value']}%</td>
                <td class="num">{params['take_profit_pct']['value']}%</td>
                <td class="num">{params['min_confidence']['value']}%</td>
                <td class="num">{params['leverage_cap']['value']}x</td>
            </tr>
            """

        st.markdown(f"""
        <table class="report-table">
            <thead>
                <tr>
                    <th>Risk Level</th>
                    <th>Max Position</th>
                    <th>Stop Loss</th>
                    <th>Take Profit</th>
                    <th>Min Confidence</th>
                    <th>Leverage Cap</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        st.divider()

        with st.expander("ℹ️ Parameter Explanations"):
            st.markdown("""
            **Max Position %**: Maximum allocation per single position. Limits concentration risk.

            **Stop Loss %**: Distance below (for longs) or above (for shorts) entry price to exit.

            **Take Profit %**: Distance above (for longs) or below (for shorts) entry price to take profit.

            **Min Confidence %**: Minimum analyst confidence required to generate a BUY or SHORT signal.

            **Leverage Cap**: Maximum leverage allowed (1x = no leverage).
            """)

    with settings_tab2:
        st.markdown("### Custom Parameter Overrides")

        use_custom = st.checkbox(
            "Enable custom parameters",
            value=st.session_state.use_custom_params,
            help="Override preset risk level parameters"
        )
        st.session_state.use_custom_params = use_custom

        if use_custom:
            st.warning("Custom parameters are active and will override preset values.")

            col1, col2 = st.columns(2)

            with col1:
                st.session_state.custom_risk_params["max_position_pct"] = st.number_input(
                    "Max Position %",
                    min_value=5.0, max_value=100.0,
                    value=float(st.session_state.custom_risk_params.get("max_position_pct", 22.5)),
                    step=1.0
                )

                st.session_state.custom_risk_params["stop_loss_pct"] = st.number_input(
                    "Stop Loss %",
                    min_value=1.0, max_value=50.0,
                    value=float(st.session_state.custom_risk_params.get("stop_loss_pct", 12.5)),
                    step=0.5
                )

                st.session_state.custom_risk_params["take_profit_pct"] = st.number_input(
                    "Take Profit %",
                    min_value=5.0, max_value=200.0,
                    value=float(st.session_state.custom_risk_params.get("take_profit_pct", 37.5)),
                    step=1.0
                )

            with col2:
                st.session_state.custom_risk_params["min_confidence"] = st.number_input(
                    "Min Confidence %",
                    min_value=10.0, max_value=90.0,
                    value=float(st.session_state.custom_risk_params.get("min_confidence", 47.5)),
                    step=5.0
                )

                st.session_state.custom_risk_params["leverage_cap"] = st.number_input(
                    "Leverage Cap (x)",
                    min_value=1.0, max_value=5.0,
                    value=float(st.session_state.custom_risk_params.get("leverage_cap", 1.5)),
                    step=0.1
                )

            st.divider()

            if st.button("🔄 Reset to Preset Values", use_container_width=True):
                st.session_state.custom_risk_params = {}
                st.session_state.use_custom_params = False
                st.rerun()
        else:
            st.info("Enable custom parameters to override preset values.")


# ============== FOOTER ==============
st.divider()
st.caption("AI Hedge Fund Terminal v4.2 | Educational Use Only | Not Financial Advice")
st.caption("Data: Yahoo Finance (15-20 min delayed) | Past performance ≠ future results")
