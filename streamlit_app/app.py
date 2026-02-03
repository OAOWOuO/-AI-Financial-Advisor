"""
AI Hedge Fund Analysis Dashboard
Streamlit version for easy sharing and deployment
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Page config
st.set_page_config(
    page_title="AI Hedge Fund",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
    }
    .signal-bullish { color: #22c55e; font-weight: bold; }
    .signal-bearish { color: #ef4444; font-weight: bold; }
    .signal-neutral { color: #6b7280; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'run_history' not in st.session_state:
    st.session_state.run_history = []

# Available analysts
ANALYSTS = [
    {"key": "warren_buffett", "name": "Warren Buffett", "desc": "Value investing principles"},
    {"key": "charlie_munger", "name": "Charlie Munger", "desc": "Mental models approach"},
    {"key": "ben_graham", "name": "Ben Graham", "desc": "Security analysis fundamentals"},
    {"key": "peter_lynch", "name": "Peter Lynch", "desc": "Growth at reasonable price"},
    {"key": "phil_fisher", "name": "Phil Fisher", "desc": "Scuttlebutt method"},
    {"key": "stanley_druckenmiller", "name": "Stanley Druckenmiller", "desc": "Macro investing"},
    {"key": "bill_ackman", "name": "Bill Ackman", "desc": "Activist investing"},
    {"key": "cathie_wood", "name": "Cathie Wood", "desc": "Disruptive innovation"},
    {"key": "fundamentals_agent", "name": "Fundamentals Agent", "desc": "Financial metrics analysis"},
    {"key": "technical_agent", "name": "Technical Agent", "desc": "Chart patterns & indicators"},
    {"key": "sentiment_agent", "name": "Sentiment Agent", "desc": "News & social sentiment"},
    {"key": "valuation_agent", "name": "Valuation Agent", "desc": "DCF & relative valuation"},
]


def run_analysis(tickers: list, analysts: list, model_name: str = "gpt-4o-mini"):
    """Run the hedge fund analysis."""
    try:
        from dotenv import load_dotenv
        load_dotenv()

        # Check for API keys - try Streamlit secrets first, then env vars
        api_key = None
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass

        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            st.error("Missing OPENAI_API_KEY. Please set it in Streamlit secrets or .env file.")
            return None

        # Set the API key in environment for the hedge fund code
        os.environ["OPENAI_API_KEY"] = api_key

        # Also try to get financial datasets key
        try:
            fin_key = st.secrets.get("FINANCIAL_DATASETS_API_KEY")
            if fin_key:
                os.environ["FINANCIAL_DATASETS_API_KEY"] = fin_key
        except:
            pass

        from src.main import run_hedge_fund

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        portfolio = {
            "cash": 100000,
            "margin_requirement": 0.0,
            "margin_used": 0.0,
            "positions": {
                ticker: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0}
                for ticker in tickers
            },
            "realized_gains": {ticker: {"long": 0.0, "short": 0.0} for ticker in tickers},
        }

        result = run_hedge_fund(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            portfolio=portfolio,
            show_reasoning=True,
            selected_analysts=analysts,
            model_name=model_name,
            model_provider="OpenAI",
        )

        return result
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None


def parse_result(result: dict, tickers: list):
    """Parse the analysis result into display format."""
    if not result:
        return None, None

    signals = result.get("analyst_signals", {})
    decisions = result.get("decisions", {})

    stocks_data = []
    for ticker in tickers:
        ticker_signals = signals.get(ticker, {})
        agents = []
        for agent_key, signal_data in ticker_signals.items():
            agents.append({
                "agent": agent_key.replace("_", " ").title(),
                "signal": signal_data.get("signal", "NEUTRAL").upper(),
                "confidence": signal_data.get("confidence", 0),
                "reasoning": str(signal_data.get("reasoning", ""))[:200],
            })

        bullish = sum(1 for a in agents if a["signal"] == "BULLISH")
        bearish = sum(1 for a in agents if a["signal"] == "BEARISH")
        neutral = sum(1 for a in agents if a["signal"] == "NEUTRAL")

        stocks_data.append({
            "ticker": ticker,
            "agents": agents,
            "bullish": bullish,
            "bearish": bearish,
            "neutral": neutral,
        })

    # Parse decisions
    portfolio_data = []
    dec_list = decisions.get("decisions", []) if isinstance(decisions, dict) else []
    for dec in dec_list:
        ticker = dec.get("ticker", "")
        ticker_signals = signals.get(ticker, {})
        bull = sum(1 for s in ticker_signals.values() if s.get("signal", "").upper() == "BULLISH")
        bear = sum(1 for s in ticker_signals.values() if s.get("signal", "").upper() == "BEARISH")
        neut = sum(1 for s in ticker_signals.values() if s.get("signal", "").upper() == "NEUTRAL")

        portfolio_data.append({
            "ticker": ticker,
            "action": dec.get("action", "hold").upper(),
            "quantity": dec.get("quantity", 0),
            "confidence": dec.get("confidence", 0),
            "bullish": bull,
            "bearish": bear,
            "neutral": neut,
        })

    return stocks_data, portfolio_data


def display_signal_badge(signal: str):
    """Return colored signal text."""
    colors = {"BULLISH": "#22c55e", "BEARISH": "#ef4444", "NEUTRAL": "#6b7280"}
    return f'<span style="color: {colors.get(signal, "#6b7280")}; font-weight: bold;">{signal}</span>'


# Sidebar
with st.sidebar:
    st.title("📈 AI Hedge Fund")
    st.markdown("---")

    # Tickers input
    st.subheader("Stock Tickers")
    ticker_input = st.text_input(
        "Enter tickers (comma-separated)",
        value="AAPL, MSFT, NVDA",
        help="Enter stock ticker symbols separated by commas"
    )
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    st.markdown("---")

    # Analyst selection
    st.subheader("AI Analysts")
    st.caption("Select which AI agents to use")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Select All", use_container_width=True):
            st.session_state.selected_analysts = [a["key"] for a in ANALYSTS]
    with col2:
        if st.button("Clear All", use_container_width=True):
            st.session_state.selected_analysts = []

    if 'selected_analysts' not in st.session_state:
        st.session_state.selected_analysts = [a["key"] for a in ANALYSTS[:6]]

    selected = []
    for analyst in ANALYSTS:
        if st.checkbox(
            analyst["name"],
            value=analyst["key"] in st.session_state.selected_analysts,
            help=analyst["desc"],
            key=f"analyst_{analyst['key']}"
        ):
            selected.append(analyst["key"])
    st.session_state.selected_analysts = selected

    st.markdown("---")

    # Model selection
    model = st.selectbox(
        "AI Model",
        ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        help="Select the OpenAI model to use"
    )

    st.markdown("---")

    # Run button
    run_disabled = len(tickers) == 0 or len(selected) == 0
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=run_disabled):
        with st.spinner("Running analysis... This may take a few minutes."):
            result = run_analysis(tickers, selected, model)
            if result:
                st.session_state.analysis_result = {
                    "result": result,
                    "tickers": tickers,
                    "timestamp": datetime.now().isoformat(),
                }
                st.session_state.run_history.append(st.session_state.analysis_result)
                st.success("Analysis complete!")
                st.rerun()

# Main content
st.title("AI Hedge Fund Analysis")

if st.session_state.analysis_result:
    result_data = st.session_state.analysis_result
    stocks_data, portfolio_data = parse_result(result_data["result"], result_data["tickers"])

    if portfolio_data:
        # Portfolio Recommendations
        st.header("📊 Portfolio Recommendations")

        cols = st.columns(len(portfolio_data))
        for i, item in enumerate(portfolio_data):
            with cols[i]:
                action_color = "#22c55e" if item["action"] in ["BUY", "LONG"] else "#ef4444" if item["action"] == "SHORT" else "#6b7280"

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {'#064e3b' if item['action'] in ['BUY', 'LONG'] else '#7f1d1d' if item['action'] == 'SHORT' else '#1f2937'} 0%, #111827 100%);
                            padding: 20px; border-radius: 12px; border: 1px solid {'#065f46' if item['action'] in ['BUY', 'LONG'] else '#991b1b' if item['action'] == 'SHORT' else '#374151'};">
                    <h2 style="margin: 0; color: white;">{item['ticker']}</h2>
                    <span style="background: {action_color}; color: white; padding: 4px 12px; border-radius: 4px; font-size: 14px; font-weight: bold;">
                        {item['action']}
                    </span>
                    <p style="margin: 15px 0 5px 0; color: #9ca3af; font-size: 12px;">Position</p>
                    <p style="margin: 0; color: white; font-size: 24px; font-weight: bold;">{item['quantity']} shares</p>
                    <p style="margin: 15px 0 5px 0; color: #9ca3af; font-size: 12px;">Confidence</p>
                    <div style="background: #374151; border-radius: 4px; height: 8px; margin-bottom: 5px;">
                        <div style="background: {'#22c55e' if item['confidence'] >= 70 else '#eab308' if item['confidence'] >= 40 else '#ef4444'};
                                    width: {item['confidence']}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <p style="margin: 0; color: white; font-size: 14px;">{item['confidence']:.0f}%</p>
                    <div style="display: flex; gap: 4px; margin-top: 15px;">
                        <div style="flex: {item['bullish']}; background: #22c55e; padding: 4px; border-radius: 4px; text-align: center; color: white; font-size: 12px;">{item['bullish']}</div>
                        <div style="flex: {max(item['neutral'], 0.1)}; background: #6b7280; padding: 4px; border-radius: 4px; text-align: center; color: white; font-size: 12px;">{item['neutral']}</div>
                        <div style="flex: {item['bearish']}; background: #ef4444; padding: 4px; border-radius: 4px; text-align: center; color: white; font-size: 12px;">{item['bearish']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")

        # Signal Summary
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Signal Summary")
            total_bull = sum(p["bullish"] for p in portfolio_data)
            total_bear = sum(p["bearish"] for p in portfolio_data)
            total_neut = sum(p["neutral"] for p in portfolio_data)

            fig = go.Figure(data=[go.Pie(
                labels=['Bullish', 'Neutral', 'Bearish'],
                values=[total_bull, total_neut, total_bear],
                hole=.4,
                marker_colors=['#22c55e', '#6b7280', '#ef4444']
            )])
            fig.update_layout(
                showlegend=True,
                height=300,
                margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🏆 Top Agents")
            if stocks_data:
                agent_stats = {}
                for stock in stocks_data:
                    for agent in stock["agents"]:
                        name = agent["agent"]
                        if name not in agent_stats:
                            agent_stats[name] = {"total": 0, "confidence": 0}
                        agent_stats[name]["total"] += 1
                        agent_stats[name]["confidence"] += agent["confidence"]

                sorted_agents = sorted(
                    [(name, stats["confidence"] / stats["total"]) for name, stats in agent_stats.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]

                for i, (name, avg_conf) in enumerate(sorted_agents):
                    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
                    st.markdown(f"""
                    <div style="display: flex; align-items: center; padding: 8px; background: #1f2937; border-radius: 8px; margin-bottom: 8px;">
                        <span style="font-size: 20px; margin-right: 10px;">{medal}</span>
                        <span style="flex: 1; color: white;">{name}</span>
                        <span style="color: #60a5fa; font-weight: bold;">{avg_conf:.0f}%</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # Detailed Analysis
        st.header("📋 Detailed Analysis")

        for stock in stocks_data:
            with st.expander(f"**{stock['ticker']}** - {stock['bullish']} Bullish | {stock['neutral']} Neutral | {stock['bearish']} Bearish"):
                df = pd.DataFrame(stock["agents"])

                for _, agent in df.iterrows():
                    signal_color = "#22c55e" if agent["signal"] == "BULLISH" else "#ef4444" if agent["signal"] == "BEARISH" else "#6b7280"
                    st.markdown(f"""
                    <div style="background: #1f2937; padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 3px solid {signal_color};">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <strong style="color: white;">{agent['agent']}</strong>
                            <span style="background: {signal_color}30; color: {signal_color}; padding: 2px 8px; border-radius: 12px; font-size: 12px;">{agent['signal']}</span>
                        </div>
                        <div style="color: #9ca3af; font-size: 12px; margin-top: 4px;">Confidence: {agent['confidence']:.0f}%</div>
                        <div style="color: #6b7280; font-size: 12px; margin-top: 8px;">{agent['reasoning'][:200]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 50px;">
        <h2>Welcome to AI Hedge Fund Analysis</h2>
        <p style="color: #9ca3af;">Select your tickers and AI analysts from the sidebar, then click <strong>Run Analysis</strong> to get started.</p>
        <br>
        <h4>How it works:</h4>
        <ol style="text-align: left; max-width: 500px; margin: 0 auto; color: #9ca3af;">
            <li>Enter stock tickers (e.g., AAPL, MSFT, NVDA)</li>
            <li>Select AI analysts to evaluate the stocks</li>
            <li>Each analyst provides a signal (Bullish/Bearish/Neutral) with confidence</li>
            <li>The portfolio manager aggregates signals into trading recommendations</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6b7280;'>Built with Streamlit | AI Hedge Fund Analysis Tool</p>",
    unsafe_allow_html=True
)
