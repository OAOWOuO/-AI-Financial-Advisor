"""
AI Financial Advisor
Combined app with session state navigation
"""

import streamlit as st

st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Navigation via URL query params (used by footer anchor links) ──────────
try:
    _nav = st.query_params.get("nav", "")
    if _nav in ("home", "portfolio", "analyzer", "caseqa"):
        st.session_state["current_view"] = _nav
        st.query_params.clear()
except Exception:
    pass

if "current_view" not in st.session_state:
    st.session_state.current_view = "home"

st.markdown("""<style>
.main { background: #0d1117; }
.stApp { background: #0d1117; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none; }
h1, h2, h3, h4 { color: #e6edf3 !important; font-weight: 600 !important; }
p, span, label, li, div { color: #c9d1d9 !important; }
.stButton > button {
    background: #238636 !important;
    color: white !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    width: 100%;
}
.stButton > button:hover { background: #2ea043 !important; }
.back-btn button {
    background: #21262d !important;
    border: 1px solid #30363d !important;
}
.tool-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
}
.tool-card:hover { border-color: #58a6ff; }
/* ─── Footer ─── */
.ft-accent {
    height: 3px;
    background: linear-gradient(90deg, #0f2a4a 0%, #1d4ed8 40%, #3b82f6 60%, #0f2a4a 100%);
    margin-top: 40px;
}
.ft-body { background: #05091a; border-top: 1px solid #0d1f38; }
/* reset all child color so footer classes can apply cleanly */
.ft-body span, .ft-body div, .ft-body a { color: inherit !important; }
.ft-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1.7fr;
    padding: 32px 48px 26px;
}
.ft-col { padding: 0 32px; }
.ft-col:first-child { padding-left: 0; }
.ft-col:last-child  { padding-right: 0; }
.ft-col:not(:last-child) { border-right: 1px solid #0d1f38; }
.ft-body .ft-label {
    display: block;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #2563eb !important;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #0d1f38;
}
.ft-body .ft-brand-name {
    display: block;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: #c8d8ea !important;
    margin-bottom: 12px;
    padding-bottom: 12px;
    border-bottom: 1px solid #0d1f38;
}
.ft-body .ft-brand-desc {
    display: block;
    font-size: 12px;
    line-height: 1.7;
    color: #6a9abf !important;
    margin-bottom: 8px;
}
.ft-body .ft-brand-sub {
    display: block;
    font-size: 11px;
    line-height: 1.7;
    color: #4a7090 !important;
}
.ft-body .ft-nav {
    display: block;
    font-size: 12.5px;
    line-height: 1;
    color: #7da8c4 !important;
    text-decoration: none !important;
    padding: 7px 0 7px 10px;
    margin-bottom: 2px;
    border-left: 2px solid transparent;
    transition: color 0.15s ease, border-left-color 0.15s ease;
}
.ft-body .ft-nav:hover {
    color: #d0e6f5 !important;
    border-left-color: #2563eb !important;
    text-decoration: none !important;
}
.ft-body .ft-data-row {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    font-size: 11.5px;
    color: #6a9abf !important;
    margin-bottom: 8px;
    line-height: 1.5;
}
.ft-body .ft-disclaimer {
    display: block;
    font-size: 10.5px;
    color: #3d6080 !important;
    line-height: 1.65;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #0a1628;
    font-style: italic;
}
.ft-copy {
    background: #030712;
    border-top: 1px solid #0a1628;
    padding: 10px 48px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 10px;
    color: #2d4a62 !important;
    letter-spacing: 0.05em;
}
.ft-copy-sep { color: #0d1f38 !important; }
</style>""", unsafe_allow_html=True)


# ============== HOME VIEW ==============
def show_home():
    st.markdown("""
<div style="text-align: center; padding: 40px 0 30px 0;">
<h1 style="font-size: 42px;">🏦 AI Financial Advisor</h1>
<p style="font-size: 18px; color: #8b949e;">Your intelligent assistant for smarter investment decisions</p>
</div>
""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""<div class="tool-card">
<div style="font-size: 48px; margin-bottom: 15px;">📊</div>
<div style="font-size: 24px; font-weight: 600; color: #e6edf3 !important;">Portfolio Allocator</div>
<p style="color: #8b949e; margin: 15px 0;">Optimize your multi-stock portfolio with AI-powered allocation, risk analytics, and rebalancing recommendations.</p>
<ul style="text-align: left; color: #8b949e; padding-left: 20px; font-size: 14px;">
<li>Multi-stock signal analysis</li>
<li>Position sizing &amp; allocation</li>
<li>Risk metrics (Sharpe, Beta, VaR)</li>
<li>Performance vs S&amp;P 500</li>
<li>Dividend income tracking</li>
<li>One-click rebalancing</li>
</ul>
</div>""", unsafe_allow_html=True)
        if st.button("Enter Portfolio Allocator →", key="btn_portfolio", use_container_width=True):
            st.session_state.current_view = "portfolio"
            st.rerun()

    with col2:
        st.markdown("""<div class="tool-card">
<div style="font-size: 48px; margin-bottom: 15px;">📈</div>
<div style="font-size: 24px; font-weight: 600; color: #e6edf3 !important;">Stock Analyzer</div>
<p style="color: #8b949e; margin: 15px 0;">Deep-dive analysis of individual stocks with technical indicators and fundamental metrics to find the best opportunities.</p>
<ul style="text-align: left; color: #8b949e; padding-left: 20px; font-size: 14px;">
<li>CFA-style technical analysis (RSI, MACD, Bollinger, ADX)</li>
<li>Fundamental scoring (valuation, profitability, growth, health)</li>
<li>Multi-model valuation (P/E, DCF, analyst consensus)</li>
<li>Return forecasts with confidence intervals</li>
<li>Support &amp; resistance levels</li>
<li>BUY / HOLD / SELL recommendation</li>
</ul>
</div>""", unsafe_allow_html=True)
        if st.button("Enter Stock Analyzer →", key="btn_analyzer", use_container_width=True):
            st.session_state.current_view = "analyzer"
            st.rerun()

    with col3:
        st.markdown("""<div class="tool-card">
<div style="font-size: 48px; margin-bottom: 15px;">📚</div>
<div style="font-size: 24px; font-weight: 600; color: #e6edf3 !important;">Case Q&amp;A</div>
<p style="color: #8b949e; margin: 15px 0;">Chat with your course materials using RAG-powered AI. Every answer is grounded in your uploaded documents with file and page citations.</p>
<ul style="text-align: left; color: #8b949e; padding-left: 20px; font-size: 14px;">
<li>Upload PDFs directly in the browser</li>
<li>Auto-indexed — no terminal needed</li>
<li>Ask any question about the materials</li>
<li>Cited answers: file + page + chunk ID</li>
<li>Refuses unsupported questions explicitly</li>
</ul>
</div>""", unsafe_allow_html=True)
        if st.button("Enter Case Q&A →", key="btn_caseqa", use_container_width=True):
            st.session_state.current_view = "caseqa"
            st.rerun()


# ============== MAIN ROUTING ==============
if st.session_state.current_view == "home":
    show_home()
elif st.session_state.current_view == "analyzer":
    from stock_analyzer import show_stock_analyzer
    show_stock_analyzer()
elif st.session_state.current_view == "portfolio":
    import os
    col_back, col_title = st.columns([1, 11])
    with col_back:
        if st.button("← Back", key="back_portfolio"):
            st.session_state.current_view = "home"
            st.rerun()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "portfolio",
        os.path.join(os.path.dirname(__file__), "portfolio_allocator.py")
    )
    portfolio_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(portfolio_module)
elif st.session_state.current_view == "caseqa":
    from case_qa import show_case_qa
    show_case_qa()


# ============== FOOTER ==============
# Rendered in the main page (not an iframe) so anchor hrefs work without sandbox restrictions.
st.markdown("""<div class="ft-accent"></div>
<div class="ft-body">
<div class="ft-grid">
<div class="ft-col">
<span class="ft-brand-name">🏦 AI FINANCIAL ADVISOR</span>
<span class="ft-brand-desc">Institutional-grade investment analytics for portfolio optimization, equity research, and document intelligence.</span>
<span class="ft-brand-sub">Built for students and practitioners seeking professional-quality financial analysis tools powered by modern AI.</span>
</div>
<div class="ft-col">
<span class="ft-label">Navigation</span>
<a class="ft-nav" href="?nav=home">Home</a>
<a class="ft-nav" href="?nav=portfolio">Portfolio Allocator</a>
<a class="ft-nav" href="?nav=analyzer">Stock Analyzer</a>
<a class="ft-nav" href="?nav=caseqa">Case Q&amp;A</a>
</div>
<div class="ft-col">
<span class="ft-label">Data &amp; Legal</span>
<div class="ft-data-row">• Market data sourced from Yahoo Finance</div>
<div class="ft-data-row">• Equity prices delayed 15–20 minutes</div>
<div class="ft-data-row">• AI signals generated via OpenAI GPT-4o</div>
<div class="ft-data-row">• Coverage restricted to US-listed equities</div>
<span class="ft-disclaimer">For educational purposes only. Not financial or investment advice. Past performance is not indicative of future results.</span>
</div>
</div>
<div class="ft-copy">
<span>© 2025 AI Financial Advisor</span>
<span class="ft-copy-sep"> | </span>
<span>Not affiliated with any registered financial institution</span>
<span class="ft-copy-sep"> | </span>
<span>For educational use only</span>
<span class="ft-copy-sep"> | </span>
<span>All rights reserved</span>
</div>
</div>""", unsafe_allow_html=True)
