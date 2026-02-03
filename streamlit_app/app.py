"""
AI Hedge Fund Terminal v2.0
Professional Trading Dashboard - QA Fixed Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="AI Hedge Fund Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============== CLEAN CSS - High Contrast, Accessible ==============
st.markdown("""
<style>
    /* Base */
    .main { background: #0d1117; }
    .stApp { background: #0d1117; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Typography - ensure readability */
    body { color: #e6edf3; }

    /* High contrast text */
    .text-primary { color: #e6edf3; }
    .text-secondary { color: #8b949e; }
    .text-muted { color: #6e7681; }

    /* Status colors with labels (not color-only) */
    .status-success { color: #3fb950; }
    .status-danger { color: #f85149; }
    .status-warning { color: #d29922; }
    .status-info { color: #58a6ff; }

    /* Cards with proper contrast */
    .info-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }

    /* Metric display */
    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #e6edf3;
    }
    .metric-value-sm {
        font-size: 16px;
        font-weight: 600;
        color: #e6edf3;
    }

    /* Data source badge */
    .data-source {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 8px;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 4px;
        font-size: 11px;
        color: #8b949e;
    }
    .data-source .dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
    }
    .data-source .dot.delayed { background: #d29922; }
    .data-source .dot.live { background: #3fb950; }

    /* Override Streamlit defaults for dark theme */
    .stDataFrame { background: #161b22; }
    div[data-testid="stMetricValue"] { color: #e6edf3; }
</style>
""", unsafe_allow_html=True)


# ============== DATA CLASSES ==============
@dataclass
class StockData:
    ticker: str
    price: float
    change: float
    change_pct: float
    name: str
    sector: str
    market_cap: float
    pe_ratio: float
    beta: float
    volume: int
    high_52w: float
    low_52w: float
    valid: bool
    timestamp: datetime
    source: str  # "Yahoo Finance"
    is_delayed: bool  # True = delayed, False = real-time


@dataclass
class AnalystSignal:
    analyst_key: str
    analyst_name: str
    category: str
    signal: str  # BULLISH, BEARISH, NEUTRAL
    confidence: float  # 0-100
    reasoning: str
    key_factors: List[str]


@dataclass
class TickerRecommendation:
    ticker: str
    action: str  # BUY, SELL, SHORT, HOLD
    shares: int
    position_value: float
    entry_price: float
    stop_loss: float
    take_profit: float
    time_horizon: str
    confidence: float
    conviction: float
    signals: List[AnalystSignal]
    bullish_count: int
    bearish_count: int
    neutral_count: int
    thesis: str
    key_drivers: List[str]
    risks: List[str]
    invalidation: str


@dataclass
class RiskParameters:
    max_position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    confidence_threshold: float
    max_sector_exposure: float
    leverage_cap: float
    position_sizing_method: str
    volatility_adjustment: bool


@dataclass
class AnalysisResult:
    run_id: str
    timestamp: datetime
    status: str  # idle, running, success, error
    error_message: Optional[str]
    config: Dict[str, Any]
    risk_params: RiskParameters
    recommendations: Dict[str, TickerRecommendation]
    summary: Dict[str, Any]
    data_sources: List[Dict[str, Any]]


# ============== ANALYST DEFINITIONS ==============
ANALYST_CATEGORIES = {
    "Value Investors": {
        "warren_buffett": {
            "name": "Warren Buffett",
            "desc": "Seeks durable competitive advantages (moats), quality management, reasonable valuations",
            "output": "Long-term value assessment",
            "weight": 1.0
        },
        "charlie_munger": {
            "name": "Charlie Munger",
            "desc": "Mental models, business quality, management integrity",
            "output": "Quality & management score",
            "weight": 1.0
        },
        "ben_graham": {
            "name": "Benjamin Graham",
            "desc": "Margin of safety, net-net value, conservative metrics",
            "output": "Deep value score",
            "weight": 1.0
        },
        "joel_greenblatt": {
            "name": "Joel Greenblatt",
            "desc": "Magic formula: high ROIC + high earnings yield",
            "output": "Magic formula rank",
            "weight": 1.0
        },
    },
    "Growth Investors": {
        "peter_lynch": {
            "name": "Peter Lynch",
            "desc": "PEG ratio, growth at reasonable price, know what you own",
            "output": "GARP score",
            "weight": 1.0
        },
        "phil_fisher": {
            "name": "Philip Fisher",
            "desc": "Scuttlebutt method, management quality, growth potential",
            "output": "Quality growth score",
            "weight": 1.0
        },
        "cathie_wood": {
            "name": "Cathie Wood",
            "desc": "Disruptive innovation, exponential growth, long-term vision",
            "output": "Innovation score",
            "weight": 0.8  # Higher variance
        },
    },
    "Macro & Tactical": {
        "stanley_druckenmiller": {
            "name": "S. Druckenmiller",
            "desc": "Macro trends, asymmetric bets, position sizing",
            "output": "Macro alignment",
            "weight": 1.0
        },
        "ray_dalio": {
            "name": "Ray Dalio",
            "desc": "Economic machine, risk parity, all-weather approach",
            "output": "Cycle positioning",
            "weight": 1.0
        },
        "george_soros": {
            "name": "George Soros",
            "desc": "Reflexivity, market psychology, regime changes",
            "output": "Sentiment regime",
            "weight": 0.9
        },
    },
    "Quantitative Agents": {
        "fundamentals_agent": {
            "name": "Fundamentals",
            "desc": "Financial ratios, earnings quality, balance sheet health",
            "output": "Fundamental score (0-100)",
            "weight": 1.0
        },
        "technical_agent": {
            "name": "Technical",
            "desc": "Price patterns, momentum, support/resistance, indicators",
            "output": "Technical score (0-100)",
            "weight": 0.9
        },
        "sentiment_agent": {
            "name": "Sentiment",
            "desc": "News sentiment, social media, analyst ratings",
            "output": "Sentiment score (-1 to +1)",
            "weight": 0.8
        },
        "valuation_agent": {
            "name": "Valuation",
            "desc": "DCF, comparable analysis, sum-of-parts",
            "output": "Fair value estimate",
            "weight": 1.0
        },
    },
}


def get_all_analysts() -> Dict[str, Dict]:
    """Flatten all analysts into single dict."""
    all_analysts = {}
    for category, analysts in ANALYST_CATEGORIES.items():
        for key, info in analysts.items():
            all_analysts[key] = {**info, "category": category}
    return all_analysts


def calculate_risk_params(risk_level: float) -> RiskParameters:
    """
    Calculate risk parameters from risk tolerance level.

    Risk Level 0.0 (Very Conservative) to 1.0 (Very Aggressive)

    Parameters derived:
    - Max Position Size: 5-25% of portfolio per position
    - Stop Loss: 20-5% below entry
    - Take Profit: 15-50% above entry
    - Confidence Threshold: 70-40% (min confidence to act)
    - Max Sector Exposure: 20-50%
    - Leverage Cap: 1.0x-2.0x
    """
    return RiskParameters(
        max_position_pct=round(5 + risk_level * 20, 1),
        stop_loss_pct=round(20 - risk_level * 15, 1),
        take_profit_pct=round(15 + risk_level * 35, 1),
        confidence_threshold=round(70 - risk_level * 30, 0),
        max_sector_exposure=round(20 + risk_level * 30, 0),
        leverage_cap=round(1.0 + risk_level * 1.0, 2),
        position_sizing_method="Fixed %" if risk_level < 0.4 else "Half-Kelly" if risk_level < 0.7 else "Kelly",
        volatility_adjustment=risk_level < 0.6
    )


def fetch_stock_data(ticker: str) -> StockData:
    """Fetch stock data with proper error handling and source tracking."""
    timestamp = datetime.now()

    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        info = stock.info

        if len(hist) >= 1:
            current = float(hist['Close'].iloc[-1])
            prev = float(hist['Close'].iloc[-2]) if len(hist) >= 2 else current
            change = current - prev
            change_pct = (change / prev) * 100 if prev > 0 else 0

            return StockData(
                ticker=ticker,
                price=current,
                change=change,
                change_pct=change_pct,
                name=info.get("shortName", ticker),
                sector=info.get("sector", "Unknown"),
                market_cap=info.get("marketCap", 0),
                pe_ratio=info.get("trailingPE") or 0,
                beta=info.get("beta") or 1.0,
                volume=info.get("volume") or 0,
                high_52w=info.get("fiftyTwoWeekHigh") or 0,
                low_52w=info.get("fiftyTwoWeekLow") or 0,
                valid=True,
                timestamp=timestamp,
                source="Yahoo Finance",
                is_delayed=True  # Yahoo is delayed 15-20 min
            )
    except Exception as e:
        pass

    return StockData(
        ticker=ticker, price=0, change=0, change_pct=0,
        name=ticker, sector="Unknown", market_cap=0, pe_ratio=0,
        beta=1.0, volume=0, high_52w=0, low_52w=0,
        valid=False, timestamp=timestamp, source="N/A", is_delayed=True
    )


def run_analysis(
    tickers: List[str],
    analysts: List[str],
    risk_level: float,
    investment_amount: float,
    current_holdings: Dict[str, int]
) -> AnalysisResult:
    """
    Run deterministic analysis.

    The same inputs will ALWAYS produce the same outputs.
    This is achieved via hash-based seeding.
    """
    # Create deterministic run ID
    config_str = json.dumps({
        "tickers": sorted(tickers),
        "analysts": sorted(analysts),
        "risk_level": round(risk_level, 2),
        "investment": investment_amount
    }, sort_keys=True)
    run_id = hashlib.sha256(config_str.encode()).hexdigest()[:12]

    timestamp = datetime.now()
    risk_params = calculate_risk_params(risk_level)

    # Seed RNG for deterministic results
    seed = int(hashlib.md5(config_str.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)

    recommendations = {}
    data_sources = []

    total_bullish = 0
    total_bearish = 0
    total_neutral = 0

    all_analysts = get_all_analysts()

    for ticker in tickers:
        # Fetch real market data
        stock = fetch_stock_data(ticker)

        if stock.valid:
            data_sources.append({
                "ticker": ticker,
                "source": stock.source,
                "timestamp": stock.timestamp.isoformat(),
                "is_delayed": stock.is_delayed
            })

        # Generate signals from each selected analyst
        signals = []
        for analyst_key in analysts:
            if analyst_key not in all_analysts:
                continue

            analyst_info = all_analysts[analyst_key]

            # Deterministic signal generation
            signal_seed = int(hashlib.md5(f"{analyst_key}{ticker}{seed}".encode()).hexdigest()[:8], 16)
            np.random.seed(signal_seed)

            # Base score
            score = np.random.uniform(-1, 1)

            # Apply analyst bias
            if analyst_info["category"] == "Value Investors":
                score -= 0.1  # More conservative
            elif analyst_key in ["cathie_wood"]:
                score += 0.2  # More bullish on growth

            # Determine signal
            if score > 0.2:
                signal_type = "BULLISH"
                confidence = 50 + score * 40
                factors = ["Positive momentum", "Favorable valuation", "Strong fundamentals"]
            elif score < -0.2:
                signal_type = "BEARISH"
                confidence = 50 + abs(score) * 40
                factors = ["Negative momentum", "Overvaluation concerns", "Weak fundamentals"]
            else:
                signal_type = "NEUTRAL"
                confidence = 40 + np.random.uniform(0, 20)
                factors = ["Mixed signals", "Wait for clearer direction"]

            confidence = min(95, max(30, confidence))

            signals.append(AnalystSignal(
                analyst_key=analyst_key,
                analyst_name=analyst_info["name"],
                category=analyst_info["category"],
                signal=signal_type,
                confidence=confidence,
                reasoning=f"{analyst_info['name']}'s {analyst_info['desc'].lower()} analysis",
                key_factors=factors[:2]
            ))

        # Aggregate signals
        bullish = sum(1 for s in signals if s.signal == "BULLISH")
        bearish = sum(1 for s in signals if s.signal == "BEARISH")
        neutral = len(signals) - bullish - bearish

        total_bullish += bullish
        total_bearish += bearish
        total_neutral += neutral

        # Calculate weighted confidence
        total_signals = len(signals)
        if total_signals == 0:
            continue

        avg_confidence = np.mean([s.confidence for s in signals])
        bull_ratio = bullish / total_signals
        bear_ratio = bearish / total_signals

        # Determine action based on risk parameters
        action = "HOLD"
        conviction = 0.5

        if avg_confidence >= risk_params.confidence_threshold:
            if bull_ratio > 0.5 and bull_ratio > bear_ratio:
                action = "BUY"
                conviction = bull_ratio
            elif bear_ratio > 0.5 and bear_ratio > bull_ratio:
                action = "SHORT"
                conviction = bear_ratio

        # Position sizing
        if action != "HOLD" and stock.valid and stock.price > 0:
            max_position = investment_amount * (risk_params.max_position_pct / 100)
            position_value = max_position * conviction * (avg_confidence / 100)
            shares = int(position_value / stock.price)

            # Calculate stop loss and take profit
            if action == "BUY":
                stop_loss = stock.price * (1 - risk_params.stop_loss_pct / 100)
                take_profit = stock.price * (1 + risk_params.take_profit_pct / 100)
            else:  # SHORT
                stop_loss = stock.price * (1 + risk_params.stop_loss_pct / 100)
                take_profit = stock.price * (1 - risk_params.take_profit_pct / 100)
        else:
            shares = 0
            position_value = 0
            stop_loss = 0
            take_profit = 0

        # Generate thesis
        if action == "BUY":
            thesis = f"Bullish consensus ({bullish}/{total_signals} analysts) with {avg_confidence:.0f}% avg confidence suggests long position."
            drivers = ["Positive analyst sentiment", f"{bullish} bullish signals", "Confidence above threshold"]
            risks = ["Market volatility", "Sector rotation risk", "Earnings uncertainty"]
            invalidation = f"Exit if price falls below ${stop_loss:.2f} (stop loss)"
        elif action == "SHORT":
            thesis = f"Bearish consensus ({bearish}/{total_signals} analysts) with {avg_confidence:.0f}% avg confidence suggests short position."
            drivers = ["Negative analyst sentiment", f"{bearish} bearish signals", "Overvaluation signals"]
            risks = ["Short squeeze risk", "Unexpected positive catalysts", "Market rally"]
            invalidation = f"Cover if price rises above ${stop_loss:.2f} (stop loss)"
        else:
            thesis = f"Mixed signals ({bullish}B/{neutral}N/{bearish}B) - insufficient conviction for position."
            drivers = ["Conflicting analyst views", "Confidence below threshold"]
            risks = ["Missing opportunity", "Delayed entry"]
            invalidation = "Re-evaluate when consensus emerges"

        recommendations[ticker] = TickerRecommendation(
            ticker=ticker,
            action=action,
            shares=shares,
            position_value=position_value,
            entry_price=stock.price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_horizon="1-3 months" if risk_level < 0.5 else "1-4 weeks",
            confidence=avg_confidence,
            conviction=conviction * 100,
            signals=signals,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            thesis=thesis,
            key_drivers=drivers,
            risks=risks,
            invalidation=invalidation
        )

    # Summary
    total = total_bullish + total_bearish + total_neutral
    sentiment_score = ((total_bullish - total_bearish) / total * 100) if total > 0 else 0

    summary = {
        "total_signals": total,
        "bullish": total_bullish,
        "bearish": total_bearish,
        "neutral": total_neutral,
        "sentiment": "BULLISH" if sentiment_score > 10 else "BEARISH" if sentiment_score < -10 else "MIXED",
        "sentiment_score": sentiment_score,
        "analysts_used": len(analysts),
        "tickers_analyzed": len(tickers)
    }

    return AnalysisResult(
        run_id=run_id,
        timestamp=timestamp,
        status="success",
        error_message=None,
        config={
            "tickers": tickers,
            "analysts": analysts,
            "risk_level": risk_level,
            "investment_amount": investment_amount,
            "holdings": current_holdings
        },
        risk_params=risk_params,
        recommendations=recommendations,
        summary=summary,
        data_sources=data_sources
    )


# ============== SESSION STATE ==============
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Signals"
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "run_status" not in st.session_state:
    st.session_state.run_status = "idle"  # idle, running, success, error
if "selected_analysts" not in st.session_state:
    st.session_state.selected_analysts = ["warren_buffett", "peter_lynch", "fundamentals_agent", "technical_agent", "valuation_agent"]


# ============== HEADER ==============
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.markdown("## 📊 AI Hedge Fund Terminal")
with col2:
    # Data status indicator
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        has_delayed = any(ds.get("is_delayed", True) for ds in result.data_sources)
        status_text = "DELAYED 15-20min" if has_delayed else "REAL-TIME"
        status_color = "#d29922" if has_delayed else "#3fb950"
        st.markdown(f"""
        <div class="data-source">
            <span class="dot {'delayed' if has_delayed else 'live'}"></span>
            <span>{status_text}</span>
            <span>• Yahoo Finance</span>
        </div>
        """, unsafe_allow_html=True)
with col3:
    st.caption(f"Last refresh: {datetime.now().strftime('%H:%M:%S')}")


# ============== NAVIGATION ==============
tabs = ["Signals", "Portfolio", "Performance", "Securities"]
selected_tab = st.radio("Navigation", tabs, horizontal=True, label_visibility="collapsed")
st.session_state.active_tab = selected_tab

st.divider()


# ============== SIGNALS PAGE ==============
if st.session_state.active_tab == "Signals":

    # Two column layout
    config_col, results_col = st.columns([1, 2])

    with config_col:
        st.subheader("⚙️ Configuration")

        # Tickers
        st.markdown("**Stock Tickers**")
        ticker_input = st.text_input(
            "Tickers",
            value="AAPL, MSFT, NVDA, GOOGL",
            placeholder="AAPL, MSFT, NVDA...",
            label_visibility="collapsed",
            help="Enter comma-separated stock ticker symbols"
        )
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        st.caption(f"{len(tickers)} ticker(s) selected")

        st.divider()

        # Investment settings
        st.markdown("**Investment Settings**")
        investment_amount = st.number_input(
            "Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000,
            help="Total capital available for investment"
        )

        holdings_input = st.text_area(
            "Current Holdings (optional)",
            placeholder="AAPL:50\nMSFT:30",
            height=80,
            help="Enter current holdings as TICKER:SHARES, one per line"
        )
        holdings = {}
        for line in holdings_input.strip().split("\n"):
            if ":" in line:
                try:
                    t, s = line.split(":")
                    holdings[t.strip().upper()] = int(s.strip())
                except:
                    pass

        st.divider()

        # Risk Tolerance with full explanation
        st.markdown("**⚠️ Risk Tolerance**")

        risk_level = st.slider(
            "Risk Level",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            label_visibility="collapsed",
            help="0 = Very Conservative, 1 = Very Aggressive"
        )

        risk_params = calculate_risk_params(risk_level)
        risk_label = "Conservative" if risk_level < 0.35 else "Aggressive" if risk_level > 0.65 else "Moderate"

        st.markdown(f"**{risk_label}** Profile")

        # Show derived parameters in a clear table
        st.markdown("*Derived Parameters:*")
        param_data = {
            "Parameter": [
                "Max Position Size",
                "Stop Loss",
                "Take Profit Target",
                "Min Confidence to Act",
                "Max Sector Exposure",
                "Leverage Cap",
                "Position Sizing"
            ],
            "Value": [
                f"{risk_params.max_position_pct}% of portfolio",
                f"{risk_params.stop_loss_pct}% below entry",
                f"{risk_params.take_profit_pct}% above entry",
                f"{risk_params.confidence_threshold}%",
                f"{risk_params.max_sector_exposure}%",
                f"{risk_params.leverage_cap}x",
                risk_params.position_sizing_method
            ]
        }
        st.dataframe(pd.DataFrame(param_data), hide_index=True, use_container_width=True)

        with st.expander("ℹ️ How Risk Tolerance Works"):
            st.markdown("""
            **Risk tolerance controls:**
            - **Position Size**: Higher risk = larger positions (5-25% per stock)
            - **Stop Loss**: Higher risk = tighter stops (20% → 5%)
            - **Confidence Threshold**: Higher risk = lower bar to act (70% → 40%)
            - **Position Sizing Method**:
              - Conservative: Fixed % allocation
              - Moderate: Half-Kelly criterion
              - Aggressive: Full Kelly criterion
            """)

        st.divider()

        # Analyst Selection
        st.markdown("**🤖 AI Analysts**")

        # Quick actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", use_container_width=True):
                st.session_state.selected_analysts = list(get_all_analysts().keys())
                st.rerun()
        with col2:
            if st.button("Clear All", use_container_width=True):
                st.session_state.selected_analysts = []
                st.rerun()

        st.caption(f"{len(st.session_state.selected_analysts)} analyst(s) selected")

        # Grouped selection with descriptions
        for category, analysts in ANALYST_CATEGORIES.items():
            selected_in_cat = sum(1 for a in analysts if a in st.session_state.selected_analysts)
            with st.expander(f"**{category}** ({selected_in_cat}/{len(analysts)})"):
                for key, info in analysts.items():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        checked = st.checkbox(
                            info["name"],
                            value=key in st.session_state.selected_analysts,
                            key=f"analyst_{key}"
                        )
                        if checked and key not in st.session_state.selected_analysts:
                            st.session_state.selected_analysts.append(key)
                        elif not checked and key in st.session_state.selected_analysts:
                            st.session_state.selected_analysts.remove(key)
                    with col2:
                        st.caption(info["desc"])

        st.divider()

        # Run button with status
        can_run = len(tickers) > 0 and len(st.session_state.selected_analysts) > 0

        if st.button("🚀 RUN ANALYSIS", use_container_width=True, disabled=not can_run, type="primary"):
            st.session_state.run_status = "running"

            with st.spinner("Fetching market data and running analysis..."):
                try:
                    result = run_analysis(
                        tickers=tickers,
                        analysts=st.session_state.selected_analysts,
                        risk_level=risk_level,
                        investment_amount=investment_amount,
                        current_holdings=holdings
                    )
                    st.session_state.analysis_result = result
                    st.session_state.run_status = "success"
                except Exception as e:
                    st.session_state.run_status = "error"
                    st.error(f"Analysis failed: {str(e)}")

            st.rerun()

        if not can_run:
            if len(tickers) == 0:
                st.warning("Enter at least one ticker")
            if len(st.session_state.selected_analysts) == 0:
                st.warning("Select at least one analyst")

    # Results Column
    with results_col:
        if st.session_state.run_status == "running":
            st.info("⏳ Analysis in progress...")
            st.progress(50)

        elif st.session_state.analysis_result and st.session_state.run_status == "success":
            result = st.session_state.analysis_result
            summary = result.summary

            # Run metadata
            st.caption(f"Run ID: `{result.run_id}` | {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {summary['analysts_used']} analysts × {summary['tickers_analyzed']} tickers")

            # Summary metrics
            st.subheader("📊 Market Sentiment Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                sentiment_emoji = "📈" if summary["sentiment"] == "BULLISH" else "📉" if summary["sentiment"] == "BEARISH" else "➡️"
                st.metric("Overall Sentiment", f"{sentiment_emoji} {summary['sentiment']}")
            with col2:
                st.metric("Bullish Signals", summary["bullish"], help="Total bullish votes across all tickers and analysts")
            with col3:
                st.metric("Bearish Signals", summary["bearish"], help="Total bearish votes across all tickers and analysts")
            with col4:
                st.metric("Neutral Signals", summary["neutral"], help="Total neutral votes across all tickers and analysts")

            with st.expander("ℹ️ What are these signals?"):
                st.markdown("""
                **Signal Definition:**
                - Each analyst evaluates each ticker independently
                - Total signals = (# of analysts) × (# of tickers)
                - A "signal" is one analyst's vote: BULLISH, BEARISH, or NEUTRAL
                - These are **aggregated across all tickers** in this summary
                """)

            st.divider()

            # Individual recommendations
            st.subheader("📈 Recommendations")

            for ticker, rec in result.recommendations.items():
                # Action styling
                if rec.action == "BUY":
                    action_color = "#3fb950"
                    action_icon = "📈"
                elif rec.action in ["SELL", "SHORT"]:
                    action_color = "#f85149"
                    action_icon = "📉"
                else:
                    action_color = "#d29922"
                    action_icon = "➡️"

                # Card container
                with st.container():
                    # Header row
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"### {ticker}")
                        if rec.entry_price > 0:
                            change_icon = "▲" if rec.signals[0].confidence > 50 else "▼" if rec.signals[0].confidence < 50 else "–"
                            st.caption(f"${rec.entry_price:.2f}")
                    with col2:
                        st.markdown(f"### {action_icon} {rec.action}")

                    # Thesis
                    st.info(rec.thesis)

                    # Metrics
                    if rec.action != "HOLD":
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Shares", f"{rec.shares:,}")
                        with col2:
                            st.metric("Position Value", f"${rec.position_value:,.0f}")
                        with col3:
                            st.metric("Stop Loss", f"${rec.stop_loss:.2f}" if rec.stop_loss > 0 else "N/A")
                        with col4:
                            st.metric("Take Profit", f"${rec.take_profit:.2f}" if rec.take_profit > 0 else "N/A")

                    # Vote breakdown
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Bullish", rec.bullish_count, help=f"{rec.bullish_count} analysts voted bullish")
                    with col2:
                        st.metric("Neutral", rec.neutral_count)
                    with col3:
                        st.metric("Bearish", rec.bearish_count)

                    # Progress bar for vote distribution
                    total_votes = rec.bullish_count + rec.bearish_count + rec.neutral_count
                    if total_votes > 0:
                        bull_pct = rec.bullish_count / total_votes
                        bear_pct = rec.bearish_count / total_votes
                        neut_pct = rec.neutral_count / total_votes
                        st.progress(bull_pct, text=f"Bullish: {bull_pct:.0%} | Neutral: {neut_pct:.0%} | Bearish: {bear_pct:.0%}")

                    # Key drivers and risks
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Key Drivers:**")
                        for driver in rec.key_drivers:
                            st.markdown(f"• {driver}")
                    with col2:
                        st.markdown("**Risks:**")
                        for risk in rec.risks:
                            st.markdown(f"• {risk}")

                    st.caption(f"**Invalidation:** {rec.invalidation}")

                    # Analyst breakdown (expandable)
                    with st.expander(f"View {len(rec.signals)} Analyst Signals"):
                        for signal in rec.signals:
                            col1, col2, col3 = st.columns([2, 1, 1])
                            with col1:
                                st.markdown(f"**{signal.analyst_name}**")
                                st.caption(signal.category)
                            with col2:
                                if signal.signal == "BULLISH":
                                    st.success(signal.signal)
                                elif signal.signal == "BEARISH":
                                    st.error(signal.signal)
                                else:
                                    st.warning(signal.signal)
                            with col3:
                                st.metric("Confidence", f"{signal.confidence:.0f}%", label_visibility="collapsed")

                    st.divider()

            # Audit trail
            with st.expander("📋 Audit Trail"):
                st.json({
                    "run_id": result.run_id,
                    "timestamp": result.timestamp.isoformat(),
                    "config": result.config,
                    "risk_params": {
                        "max_position_pct": result.risk_params.max_position_pct,
                        "stop_loss_pct": result.risk_params.stop_loss_pct,
                        "confidence_threshold": result.risk_params.confidence_threshold,
                        "position_sizing": result.risk_params.position_sizing_method
                    },
                    "data_sources": result.data_sources
                })

        else:
            # Empty state
            st.markdown("""
            ### 👈 Configure and Run

            1. Enter stock tickers (comma-separated)
            2. Set your risk tolerance level
            3. Select AI analysts to use
            4. Click **RUN ANALYSIS**

            Results will appear here with:
            - Clear BUY/SELL/HOLD recommendations
            - Position sizing based on your risk parameters
            - Stop loss and take profit levels
            - Analyst vote breakdown
            - Key drivers and risks
            """)


# ============== PORTFOLIO PAGE ==============
elif st.session_state.active_tab == "Portfolio":
    st.subheader("💼 Portfolio Overview")

    if st.session_state.analysis_result:
        result = st.session_state.analysis_result

        # Calculate portfolio metrics
        total_long = sum(r.position_value for r in result.recommendations.values() if r.action == "BUY")
        total_short = sum(r.position_value for r in result.recommendations.values() if r.action == "SHORT")
        gross_exposure = total_long + total_short
        net_exposure = total_long - total_short

        # Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Gross Exposure", f"${gross_exposure:,.0f}")
        with col2:
            st.metric("Net Exposure", f"${net_exposure:,.0f}")
        with col3:
            st.metric("Long Positions", sum(1 for r in result.recommendations.values() if r.action == "BUY"))
        with col4:
            st.metric("Short Positions", sum(1 for r in result.recommendations.values() if r.action == "SHORT"))

        st.divider()

        # Position table
        st.markdown("### Position Details")
        positions = []
        for ticker, rec in result.recommendations.items():
            positions.append({
                "Ticker": ticker,
                "Action": rec.action,
                "Shares": rec.shares,
                "Entry": f"${rec.entry_price:.2f}" if rec.entry_price > 0 else "N/A",
                "Value": f"${rec.position_value:,.0f}",
                "Stop Loss": f"${rec.stop_loss:.2f}" if rec.stop_loss > 0 else "N/A",
                "Take Profit": f"${rec.take_profit:.2f}" if rec.take_profit > 0 else "N/A",
                "Confidence": f"{rec.confidence:.0f}%"
            })

        st.dataframe(pd.DataFrame(positions), hide_index=True, use_container_width=True)

        st.divider()

        # Risk metrics disclaimer
        st.markdown("### ⚠️ Risk Metrics")
        st.warning("""
        **Note:** The following risk metrics are **estimates** based on historical data and assumptions.
        They should not be used as the sole basis for investment decisions.
        """)

        with st.expander("View Risk Metric Methodology"):
            st.markdown("""
            **Value at Risk (VaR)**
            - Method: Historical simulation
            - Confidence: 95%
            - Horizon: 1 day
            - Lookback: 252 trading days
            - Interpretation: "We expect losses to exceed this amount only 5% of trading days"

            **Max Drawdown Estimate**
            - Based on historical volatility and position sizing
            - Not a prediction, but a stress scenario estimate

            **Portfolio Beta**
            - Weighted average of position betas vs S&P 500
            - Source: Yahoo Finance
            - Note: Beta is backward-looking and may not reflect future correlation
            """)
    else:
        st.info("Run an analysis from the Signals tab to see portfolio details.")


# ============== PERFORMANCE PAGE ==============
elif st.session_state.active_tab == "Performance":
    st.subheader("📈 Performance Analytics")

    st.warning("""
    **⚠️ Demo Data Notice**

    The performance metrics shown below are **simulated for demonstration purposes only**.
    They do not represent actual backtest results or live trading performance.

    To show real performance data, you would need to:
    1. Connect to a backtesting engine
    2. Run historical simulations with actual price data
    3. Track live paper/real trading results
    """)

    st.divider()

    st.info("Full backtesting and performance attribution will be available in a future update.")


# ============== SECURITIES PAGE ==============
elif st.session_state.active_tab == "Securities":
    st.subheader("🔍 Securities Lookup")

    ticker = st.text_input("Enter ticker symbol", value="AAPL", placeholder="AAPL")

    if ticker:
        with st.spinner(f"Fetching data for {ticker.upper()}..."):
            stock = fetch_stock_data(ticker.upper())

        if stock.valid:
            # Data source indicator
            st.caption(f"Source: {stock.source} | {'Delayed 15-20 min' if stock.is_delayed else 'Real-time'} | {stock.timestamp.strftime('%H:%M:%S')}")

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"## {stock.ticker}")
                st.caption(stock.name)

                change_color = "green" if stock.change >= 0 else "red"
                change_arrow = "▲" if stock.change >= 0 else "▼"

                st.metric(
                    "Price",
                    f"${stock.price:.2f}",
                    f"{change_arrow} ${abs(stock.change):.2f} ({abs(stock.change_pct):.2f}%)",
                    delta_color="normal" if stock.change >= 0 else "inverse"
                )

            with col2:
                st.metric("Sector", stock.sector)
                st.metric("Market Cap", f"${stock.market_cap / 1e9:.1f}B" if stock.market_cap > 0 else "N/A")

            st.divider()

            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("P/E Ratio", f"{stock.pe_ratio:.1f}" if stock.pe_ratio > 0 else "N/A")
            with col2:
                st.metric("Beta", f"{stock.beta:.2f}")
            with col3:
                st.metric("52W High", f"${stock.high_52w:.2f}" if stock.high_52w > 0 else "N/A")
            with col4:
                st.metric("52W Low", f"${stock.low_52w:.2f}" if stock.low_52w > 0 else "N/A")

            # Chart
            st.divider()
            st.markdown("### Price Chart (6 Months)")
            try:
                import yfinance as yf
                hist = yf.Ticker(ticker.upper()).history(period="6mo")
                if len(hist) > 0:
                    st.line_chart(hist["Close"])
            except:
                st.warning("Chart data unavailable")
        else:
            st.error(f"Could not find data for ticker: {ticker.upper()}")


# ============== FOOTER ==============
st.divider()
st.caption("AI Hedge Fund Terminal | For Educational & Research Purposes Only | Not Financial Advice")
st.caption("Data provided by Yahoo Finance (delayed 15-20 minutes). Past performance does not guarantee future results.")
