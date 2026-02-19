"""
AI Financial Advisor
Combined app with session state navigation
"""

import streamlit as st

st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="\U0001f3e6",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# \u2500\u2500 Keyboard-shortcut navigation via URL query params \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
try:
    _nav = st.query_params.get("nav", "")
    if _nav in ("home", "portfolio", "analyzer", "caseqa"):
        st.session_state["current_view"] = _nav
        st.query_params.clear()
except Exception:
    pass

# Initialize session state for navigation
if "current_view" not in st.session_state:
    st.session_state.current_view = "home"

# CSS
st.markdown("""
<style>
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

    /* Ensure content area is tall enough to push footer down */
    .main .block-container {
        min-height: calc(100vh - 380px);
        padding-bottom: 0 !important;
    }

    /* ====== FULL FOOTER ====== */
    .aifa-footer-wrap { margin-top: 64px; }

    .aifa-footer-full {
        background: #010409;
        border-top: 2px solid #21262d;
    }

    .aifa-footer-inner {
        display: grid;
        grid-template-columns: 2fr 1.1fr 1.3fr 1.5fr;
        gap: 48px;
        padding: 44px 48px 36px;
    }

    /* Column header */
    .aifa-footer-col-hdr {
        font-family: 'SF Mono', 'Consolas', 'Liberation Mono', monospace;
        font-size: 10px;
        font-weight: 700;
        color: #58a6ff !important;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 18px;
        padding-bottom: 8px;
        border-bottom: 1px solid #21262d;
        display: block;
    }

    /* Brand column */
    .aifa-footer-logo {
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 15px;
        font-weight: 700;
        color: #e6edf3 !important;
        letter-spacing: 0.08em;
        margin-bottom: 14px;
        display: block;
    }
    .aifa-footer-tagline {
        color: #8b949e !important;
        font-size: 13px;
        line-height: 1.65;
        margin-bottom: 10px;
        display: block;
    }
    .aifa-footer-sub {
        color: #484f58 !important;
        font-size: 12px;
        line-height: 1.65;
        display: block;
    }

    /* Navigation buttons */
    .aifa-nav-btn {
        display: block;
        padding: 8px 12px;
        margin-bottom: 5px;
        color: #8b949e !important;
        text-decoration: none !important;
        font-size: 13px;
        border-radius: 5px;
        border: 1px solid transparent;
        cursor: pointer;
    }
    .aifa-nav-btn:hover {
        color: #e6edf3 !important;
        background: #161b22 !important;
        border-color: #30363d !important;
        text-decoration: none !important;
    }

    /* Keyboard shortcut rows */
    .aifa-shortcut-row {
        display: flex;
        align-items: center;
        gap: 4px;
        margin-bottom: 10px;
    }
    .aifa-kbd {
        display: inline-block;
        background: #161b22;
        border: 1px solid #30363d;
        border-bottom: 2px solid #21262d;
        border-radius: 3px;
        padding: 1px 6px;
        font-size: 10px;
        color: #58a6ff !important;
        font-family: 'SF Mono', 'Consolas', monospace;
        letter-spacing: 0.04em;
        line-height: 1.6;
    }
    .aifa-shortcut-label {
        color: #6e7681 !important;
        font-size: 12px;
        margin-left: 6px;
    }

    /* Data & legal column */
    .aifa-footer-data-row {
        font-size: 12px;
        color: #6e7681 !important;
        margin-bottom: 9px;
        line-height: 1.5;
        display: block;
    }
    .aifa-footer-disclaimer {
        color: #484f58 !important;
        font-size: 11px;
        font-style: italic;
        line-height: 1.7;
        margin-top: 14px;
        padding-top: 14px;
        border-top: 1px solid #161b22;
        display: block;
    }

    /* Bottom copyright strip */
    .aifa-footer-bottom-bar {
        background: #000000;
        border-top: 1px solid #161b22;
        padding: 11px 48px;
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        font-size: 11px;
        color: #484f58 !important;
        font-family: 'SF Mono', 'Consolas', monospace;
        letter-spacing: 0.02em;
    }
    .aifa-bottom-sep {
        color: #21262d !important;
    }
</style>
""", unsafe_allow_html=True)


# ============== HOME VIEW ==============
def show_home():
    st.markdown("""
    <div style="text-align: center; padding: 40px 0 30px 0;">
        <h1 style="font-size: 42px;">\U0001f3e6 AI Financial Advisor</h1>
        <p style="font-size: 18px; color: #8b949e;">Your intelligent assistant for smarter investment decisions</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="large")

    with col1:
        st.markdown("""
        <div class="tool-card">
            <div style="font-size: 48px; margin-bottom: 15px;">\U0001f4ca</div>
            <div style="font-size: 24px; font-weight: 600; color: #e6edf3 !important;">Portfolio Allocator</div>
            <p style="color: #8b949e; margin: 15px 0;">
                Optimize your multi-stock portfolio with AI-powered allocation,
                risk analytics, and rebalancing recommendations.
            </p>
            <ul style="text-align: left; color: #8b949e; padding-left: 20px; font-size: 14px;">
                <li>Multi-stock signal analysis</li>
                <li>Position sizing &amp; allocation</li>
                <li>Risk metrics (Sharpe, Beta, VaR)</li>
                <li>Performance vs S&amp;P 500</li>
                <li>Dividend income tracking</li>
                <li>One-click rebalancing</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Portfolio Allocator \u2192", key="btn_portfolio", use_container_width=True):
            st.session_state.current_view = "portfolio"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="tool-card">
            <div style="font-size: 48px; margin-bottom: 15px;">\U0001f4c8</div>
            <div style="font-size: 24px; font-weight: 600; color: #e6edf3 !important;">Stock Analyzer</div>
            <p style="color: #8b949e; margin: 15px 0;">
                Deep-dive analysis of individual stocks with technical indicators
                and fundamental metrics to find the best opportunities.
            </p>
            <ul style="text-align: left; color: #8b949e; padding-left: 20px; font-size: 14px;">
                <li>CFA-style technical analysis (RSI, MACD, Bollinger, ADX)</li>
                <li>Fundamental scoring (valuation, profitability, growth, health)</li>
                <li>Multi-model valuation (P/E, DCF, analyst consensus)</li>
                <li>Return forecasts with confidence intervals</li>
                <li>Support &amp; resistance levels</li>
                <li>BUY / HOLD / SELL recommendation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Stock Analyzer \u2192", key="btn_analyzer", use_container_width=True):
            st.session_state.current_view = "analyzer"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="tool-card">
            <div style="font-size: 48px; margin-bottom: 15px;">\U0001f4da</div>
            <div style="font-size: 24px; font-weight: 600; color: #e6edf3 !important;">Case Q&amp;A</div>
            <p style="color: #8b949e; margin: 15px 0;">
                Chat with your course materials using RAG-powered AI.
                Every answer is grounded in your uploaded documents
                with file and page citations.
            </p>
            <ul style="text-align: left; color: #8b949e; padding-left: 20px; font-size: 14px;">
                <li>Upload PDFs directly in the browser</li>
                <li>Auto-indexed \u2014 no terminal needed</li>
                <li>Ask any question about the materials</li>
                <li>Cited answers: file + page + chunk ID</li>
                <li>Refuses unsupported questions explicitly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Enter Case Q&A \u2192", key="btn_caseqa", use_container_width=True):
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
        if st.button("\u2190 Back", key="back_portfolio"):
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


# ============== FOOTER (rendered on every view) ==============
st.markdown("""
<div class="aifa-footer-wrap">
  <div class="aifa-footer-full">

    <div class="aifa-footer-inner">

      <!-- Column 1: Brand -->
      <div>
        <span class="aifa-footer-logo">\U0001f3e6&nbsp; AI FINANCIAL ADVISOR</span>
        <span class="aifa-footer-tagline">
          Intelligent investment analytics powered by artificial intelligence.
        </span>
        <span class="aifa-footer-sub">
          Professional-grade tools for portfolio optimization, equity analysis,
          and document intelligence &mdash; built for students and practitioners.
        </span>
      </div>

      <!-- Column 2: Navigation -->
      <div>
        <span class="aifa-footer-col-hdr">Navigation</span>
        <a class="aifa-nav-btn" data-nav="home" href="#">\U0001f3e0&nbsp;&nbsp;Home</a>
        <a class="aifa-nav-btn" data-nav="portfolio" href="#">\U0001f4ca&nbsp;&nbsp;Portfolio Allocator</a>
        <a class="aifa-nav-btn" data-nav="analyzer" href="#">\U0001f4c8&nbsp;&nbsp;Stock Analyzer</a>
        <a class="aifa-nav-btn" data-nav="caseqa" href="#">\U0001f4da&nbsp;&nbsp;Case Q&amp;A</a>
      </div>

      <!-- Column 3: Keyboard shortcuts -->
      <div>
        <span class="aifa-footer-col-hdr">Keyboard Shortcuts</span>
        <div class="aifa-shortcut-row">
          <span class="aifa-kbd">Alt</span><span class="aifa-kbd">H</span>
          <span class="aifa-shortcut-label">Home</span>
        </div>
        <div class="aifa-shortcut-row">
          <span class="aifa-kbd">Alt</span><span class="aifa-kbd">P</span>
          <span class="aifa-shortcut-label">Portfolio Allocator</span>
        </div>
        <div class="aifa-shortcut-row">
          <span class="aifa-kbd">Alt</span><span class="aifa-kbd">S</span>
          <span class="aifa-shortcut-label">Stock Analyzer</span>
        </div>
        <div class="aifa-shortcut-row">
          <span class="aifa-kbd">Alt</span><span class="aifa-kbd">C</span>
          <span class="aifa-shortcut-label">Case Q&amp;A</span>
        </div>
      </div>

      <!-- Column 4: Data & Legal -->
      <div>
        <span class="aifa-footer-col-hdr">Data &amp; Legal</span>
        <span class="aifa-footer-data-row">\U0001f4e1&nbsp; Market data via Yahoo Finance</span>
        <span class="aifa-footer-data-row">\u23f1\ufe0f&nbsp; Prices delayed 15&ndash;20 minutes</span>
        <span class="aifa-footer-data-row">\U0001f916&nbsp; AI analysis via OpenAI GPT-4o</span>
        <span class="aifa-footer-data-row">\U0001f310&nbsp; Coverage: US-listed equities</span>
        <span class="aifa-footer-disclaimer">
          This platform is for educational purposes only and does not constitute
          financial, investment, or legal advice. Past performance is not
          indicative of future results.
        </span>
      </div>

    </div>

    <!-- Copyright strip -->
    <div class="aifa-footer-bottom-bar">
      <span>&copy; 2025 AI Financial Advisor</span>
      <span class="aifa-bottom-sep">&middot;</span>
      <span>Not affiliated with any financial institution</span>
      <span class="aifa-bottom-sep">&middot;</span>
      <span>For educational use only</span>
      <span class="aifa-bottom-sep">&middot;</span>
      <span>All rights reserved</span>
    </div>

  </div>
</div>
""", unsafe_allow_html=True)

# Navigation JS: footer button clicks + keyboard shortcuts
import streamlit.components.v1 as components
components.html("""
<script>
(function () {
  function goTo(view) {
    try {
      var url = new URL(window.parent.location.href);
      url.searchParams.set('nav', view);
      window.parent.location.href = url.toString();
    } catch (e) {}
  }

  // Wire footer nav buttons after DOM settles
  function hookFooterNav() {
    var btns = window.parent.document.querySelectorAll('.aifa-nav-btn[data-nav]');
    btns.forEach(function (el) {
      el.onclick = function (e) {
        e.preventDefault();
        goTo(el.getAttribute('data-nav'));
      };
    });
  }
  setTimeout(hookFooterNav, 300);

  // Keyboard shortcuts: Alt + H / P / S / C
  window.parent.document.addEventListener('keydown', function (e) {
    if (e.altKey && !e.ctrlKey && !e.metaKey) {
      switch (e.key.toLowerCase()) {
        case 'h': e.preventDefault(); goTo('home'); break;
        case 'p': e.preventDefault(); goTo('portfolio'); break;
        case 's': e.preventDefault(); goTo('analyzer'); break;
        case 'c': e.preventDefault(); goTo('caseqa'); break;
      }
    }
  });
})();
</script>
""", height=0)
