"""
AI Financial Advisor - Landing Page
Your intelligent assistant for portfolio allocation and stock analysis
"""

import streamlit as st

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============== CSS ==============
st.markdown("""
<style>
    /* Base dark theme */
    .main { background: #0d1117; }
    .stApp { background: #0d1117; }
    #MainMenu, footer, header { visibility: hidden; }

    /* Hide default sidebar nav for landing page */
    [data-testid="stSidebarNav"] { display: none; }

    /* Text styling */
    h1, h2, h3, h4 { color: #e6edf3 !important; font-weight: 600 !important; }
    p, span, label, li, div { color: #c9d1d9 !important; }

    /* Card styling */
    .project-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        transition: all 0.3s ease;
        height: 100%;
    }
    .project-card:hover {
        border-color: #58a6ff;
        transform: translateY(-2px);
    }
    .card-icon {
        font-size: 48px;
        margin-bottom: 15px;
    }
    .card-title {
        font-size: 24px;
        font-weight: 600;
        color: #e6edf3 !important;
        margin-bottom: 10px;
    }
    .card-desc {
        color: #8b949e !important;
        font-size: 14px;
        line-height: 1.6;
    }
    .feature-list {
        text-align: left;
        margin-top: 20px;
        padding-left: 20px;
    }
    .feature-list li {
        color: #8b949e !important;
        margin-bottom: 8px;
        font-size: 13px;
    }

    /* Button styling */
    .stButton > button {
        background: #238636 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        width: 100%;
        margin-top: 20px;
    }
    .stButton > button:hover {
        background: #2ea043 !important;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 40px 0 50px 0;
    }
    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #e6edf3 !important;
        margin-bottom: 10px;
    }
    .main-subtitle {
        font-size: 18px;
        color: #8b949e !important;
    }
</style>
""", unsafe_allow_html=True)

# ============== HEADER ==============
st.markdown("""
<div class="main-header">
    <div class="main-title">🏦 AI Financial Advisor</div>
    <div class="main-subtitle">Your intelligent assistant for smarter investment decisions</div>
</div>
""", unsafe_allow_html=True)

# ============== PROJECT CARDS ==============
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="project-card">
        <div class="card-icon">📊</div>
        <div class="card-title">Portfolio Allocator</div>
        <div class="card-desc">
            Optimize your multi-stock portfolio with AI-powered allocation,
            risk analytics, and rebalancing recommendations.
        </div>
        <ul class="feature-list">
            <li>Multi-stock signal analysis</li>
            <li>Position sizing & allocation</li>
            <li>Risk metrics (Sharpe, Beta, VaR)</li>
            <li>Performance vs S&P 500</li>
            <li>Dividend income tracking</li>
            <li>One-click rebalancing</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Enter Portfolio Allocator →", key="portfolio_btn"):
        st.switch_page("pages/1_📊_Portfolio_Allocator.py")

with col2:
    st.markdown("""
    <div class="project-card">
        <div class="card-icon">📈</div>
        <div class="card-title">Stock Analyzer</div>
        <div class="card-desc">
            Deep-dive analysis of individual stocks with technical indicators
            and fundamental metrics to find the best opportunities.
        </div>
        <ul class="feature-list">
            <li>Candlestick & volume charts</li>
            <li>Technical indicators (RSI, MACD, MA)</li>
            <li>Support & resistance levels</li>
            <li>Fundamental ratios (P/E, P/B, P/S)</li>
            <li>Growth & profitability metrics</li>
            <li>AI recommendation engine</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Enter Stock Analyzer →", key="analyzer_btn"):
        st.switch_page("pages/2_📈_Stock_Analyzer.py")

# ============== FOOTER ==============
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #6e7681; font-size: 12px; padding: 20px;">
    Data provided by Yahoo Finance (15-20 min delayed) • For educational purposes only • Not financial advice
</div>
""", unsafe_allow_html=True)
