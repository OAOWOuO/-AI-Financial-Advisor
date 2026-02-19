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

if "current_view" not in st.session_state:
    st.session_state.current_view = "home"

st.markdown("""<style>
.main { background: #0d1117; }
.stApp { background: #0d1117; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebarNav"] { display: none; }
h1, h2, h3, h4 { color: #e6edf3 !important; font-weight: 600 !important; }
p, span, label, li, div { color: #c9d1d9 !important; }

/* ── Entry buttons: institutional blue ── */
.stButton > button {
    background: #1d4ed8 !important;
    color: #e8f0fe !important;
    border: 1px solid #2563eb !important;
    border-radius: 6px !important;
    padding: 12px 24px !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    width: 100%;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    background: #2563eb !important;
    border-color: #3b82f6 !important;
    color: #fff !important;
}
.stButton > button[kind="secondary"] {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    font-size: 14px !important;
    padding: 8px 16px !important;
}
.stButton > button[kind="secondary"]:hover {
    background: #21262d !important;
    border-color: #58a6ff !important;
    color: #e6edf3 !important;
}

.tool-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
}
.tool-card:hover { border-color: #58a6ff; }

/* ═══════════════════════════════════════
   FOOTER
   ═══════════════════════════════════════

   Structure (Python side):
     with st.container():          ← creates a nested stVerticalBlock
       st.markdown(ft-accent)
       st.columns([...])           ← stHorizontalBlock with .ft-nav-section
       st.markdown(ft-copy)

   Key selector:
     [stVerticalBlock]:has(> element-container > stHorizontalBlock:has(.ft-nav-section))
   … matches ONLY the container's inner stVerticalBlock (direct child chain),
   NOT the outer page stVerticalBlock (whose direct child element-container
   wraps another stVerticalBlock, not the stHorizontalBlock directly).
   Setting gap:0 on it collapses all three footer sections flush together.
*/

/* 1 — Close inter-section gaps inside the footer container */
[data-testid="stVerticalBlock"]:has(
  > [data-testid="element-container"]
  > [data-testid="stHorizontalBlock"]:has(.ft-nav-section)
) {
    gap: 0 !important;
}

/* 2 — Accent bar */
.ft-accent {
    height: 3px;
    background: linear-gradient(90deg, #0f2a4a 0%, #1d4ed8 40%, #3b82f6 60%, #0f2a4a 100%);
    margin-top: 48px;
}
[data-testid="element-container"]:has(.ft-accent) {
    padding: 0 !important;
    margin-bottom: 0 !important;
    line-height: 0;
}

/* 3 — Footer columns row */
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) {
    background: #05091a !important;
    border-top: 1px solid #0d1f38;
    gap: 0 !important;
    align-items: stretch !important;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"] {
    padding: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:not(:last-child) {
    border-right: 1px solid #0d1f38;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"] > [data-testid="stVerticalBlock"] {
    padding: 32px 28px !important;
    background: #05091a !important;
    gap: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:first-child > [data-testid="stVerticalBlock"] {
    padding-left: 48px !important;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:last-child > [data-testid="stVerticalBlock"] {
    padding-right: 48px !important;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) [data-testid="element-container"] {
    padding: 0 !important;
    margin: 0 !important;
    width: 100% !important;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) .stMarkdownContainer {
    margin: 0 !important;
    padding: 0 !important;
}
[data-testid="stVerticalBlock"]:has(.ft-nav-section) .stButton {
    margin: 0 !important;
    padding: 0 !important;
    width: 100% !important;
}

/* Reset global text-color override inside footer */
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) span,
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) div,
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) p { color: inherit !important; }

/* 4 — Footer text elements */
.ft-label {
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
.ft-brand-name {
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
.ft-brand-desc {
    display: block;
    font-size: 12px;
    line-height: 1.7;
    color: #6a9abf !important;
    margin-bottom: 8px;
}
.ft-brand-sub {
    display: block;
    font-size: 11px;
    line-height: 1.7;
    color: #4a7090 !important;
}
.ft-data-row {
    font-size: 11.5px;
    color: #6a9abf !important;
    margin-bottom: 8px;
    line-height: 1.5;
}
.ft-disclaimer {
    display: block;
    font-size: 10.5px;
    color: #3d6080 !important;
    line-height: 1.65;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid #0a1628;
    font-style: italic;
}

/* 5 — Nav buttons scoped to navigation column */
[data-testid="stVerticalBlock"]:has(.ft-nav-section) .stButton > button {
    background: transparent !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    color: #7da8c4 !important;
    font-size: 12.5px !important;
    font-weight: 400 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 8px 0 8px 10px !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1 !important;
    border-radius: 0 !important;
    width: 100% !important;
    box-shadow: none !important;
    letter-spacing: 0.01em !important;
}
[data-testid="stVerticalBlock"]:has(.ft-nav-section) .stButton > button:hover {
    background: rgba(37,99,235,0.07) !important;
    color: #d0e6f5 !important;
    border-left: 2px solid #2563eb !important;
}
[data-testid="stVerticalBlock"]:has(.ft-nav-section) .stButton > button:focus:not(:active) {
    box-shadow: none !important;
    border-left: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stVerticalBlock"]:has(.ft-nav-section) .stButton > button:active {
    background: rgba(37,99,235,0.12) !important;
    color: #d0e6f5 !important;
}

/* 6 — Copyright strip */
.ft-copy {
    background: #030712;
    border-top: 1px solid #0a1628;
    padding: 11px 48px;
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 10px;
    color: #2d4a62 !important;
    letter-spacing: 0.05em;
}
[data-testid="element-container"]:has(.ft-copy) {
    padding: 0 !important;
    margin: 0 !important;
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
# Wrapped in st.container() so the three sections (accent · columns · copyright)
# share a single nested stVerticalBlock — targeted above to collapse its gap to 0.
with st.container():
    st.markdown('<div class="ft-accent"></div>', unsafe_allow_html=True)

    ft_col1, ft_col2, ft_col3 = st.columns([2, 1, 1.7])

    with ft_col1:
        st.markdown("""<span class="ft-brand-name">🏦 AI FINANCIAL ADVISOR</span>
<span class="ft-brand-desc">Institutional-grade investment analytics for portfolio optimization, equity research, and document intelligence.</span>
<span class="ft-brand-sub">Built for students and practitioners seeking professional-quality financial analysis tools powered by modern AI.</span>""", unsafe_allow_html=True)

    with ft_col2:
        # .ft-nav-section marker scopes nav button styles to this column only
        st.markdown('<span class="ft-label">Navigation</span><div class="ft-nav-section"></div>', unsafe_allow_html=True)
        if st.button("Home", key="ft_home"):
            st.session_state.current_view = "home"
            st.rerun()
        if st.button("Portfolio Allocator", key="ft_portfolio"):
            st.session_state.current_view = "portfolio"
            st.rerun()
        if st.button("Stock Analyzer", key="ft_analyzer"):
            st.session_state.current_view = "analyzer"
            st.rerun()
        if st.button("Case Q&A", key="ft_caseqa"):
            st.session_state.current_view = "caseqa"
            st.rerun()

    with ft_col3:
        st.markdown("""<span class="ft-label">Data &amp; Legal</span>
<div class="ft-data-row">• Market data sourced from Yahoo Finance</div>
<div class="ft-data-row">• Equity prices delayed 15–20 minutes</div>
<div class="ft-data-row">• AI signals generated via OpenAI GPT-4o</div>
<div class="ft-data-row">• Coverage restricted to US-listed equities</div>
<span class="ft-disclaimer">For educational purposes only. Not financial or investment advice. Past performance is not indicative of future results.</span>""", unsafe_allow_html=True)

    st.markdown("""<div class="ft-copy">
<span>© 2025 AI Financial Advisor</span>
<span class="ft-copy-sep"> | </span>
<span>Not affiliated with any registered financial institution</span>
<span class="ft-copy-sep"> | </span>
<span>For educational use only</span>
<span class="ft-copy-sep"> | </span>
<span>All rights reserved</span>
</div>""", unsafe_allow_html=True)
