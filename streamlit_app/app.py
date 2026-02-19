"""
AI Financial Advisor
Combined app with session state navigation
"""

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="AI Financial Advisor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Keyboard-shortcut navigation via URL query params ──────────────────────
try:
    _nav = st.query_params.get("nav", "")
    if _nav in ("home", "portfolio", "analyzer", "caseqa"):
        st.session_state["current_view"] = _nav
        st.query_params.clear()
except Exception:
    pass

if "current_view" not in st.session_state:
    st.session_state.current_view = "home"

# CSS (app styles only — no footer CSS here)
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
</style>
""", unsafe_allow_html=True)


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
        st.markdown("""
        <div class="tool-card">
            <div style="font-size: 48px; margin-bottom: 15px;">📊</div>
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
        if st.button("Enter Portfolio Allocator →", key="btn_portfolio", use_container_width=True):
            st.session_state.current_view = "portfolio"
            st.rerun()

    with col2:
        st.markdown("""
        <div class="tool-card">
            <div style="font-size: 48px; margin-bottom: 15px;">📈</div>
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
        if st.button("Enter Stock Analyzer →", key="btn_analyzer", use_container_width=True):
            st.session_state.current_view = "analyzer"
            st.rerun()

    with col3:
        st.markdown("""
        <div class="tool-card">
            <div style="font-size: 48px; margin-bottom: 15px;">📚</div>
            <div style="font-size: 24px; font-weight: 600; color: #e6edf3 !important;">Case Q&amp;A</div>
            <p style="color: #8b949e; margin: 15px 0;">
                Chat with your course materials using RAG-powered AI.
                Every answer is grounded in your uploaded documents
                with file and page citations.
            </p>
            <ul style="text-align: left; color: #8b949e; padding-left: 20px; font-size: 14px;">
                <li>Upload PDFs directly in the browser</li>
                <li>Auto-indexed — no terminal needed</li>
                <li>Ask any question about the materials</li>
                <li>Cited answers: file + page + chunk ID</li>
                <li>Refuses unsupported questions explicitly</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
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


# ============== FOOTER ──────────────────────────────────────────────────────
# Rendered entirely inside components.html so the HTML/CSS/JS are never
# touched by Streamlit's markdown parser.
components.html("""
<!DOCTYPE html>
<html>
<head>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { background: transparent; overflow: hidden; font-size: 14px; }

  .ft {
    background: #010409;
    border-top: 2px solid #21262d;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, sans-serif;
  }

  /* Four-column grid */
  .ft-inner {
    display: grid;
    grid-template-columns: 2fr 1.1fr 1.3fr 1.5fr;
    gap: 48px;
    padding: 36px 48px 28px;
  }

  /* Column header */
  .ft-hdr {
    display: block;
    font-family: 'SF Mono', 'Consolas', 'Liberation Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    color: #58a6ff;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
  }

  /* Brand column */
  .ft-logo {
    display: block;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 14px;
    font-weight: 700;
    color: #e6edf3;
    letter-spacing: 0.08em;
    margin-bottom: 12px;
  }
  .ft-tagline {
    display: block;
    color: #8b949e;
    font-size: 12px;
    line-height: 1.65;
    margin-bottom: 8px;
  }
  .ft-sub {
    display: block;
    color: #484f58;
    font-size: 11px;
    line-height: 1.65;
  }

  /* Nav buttons */
  .ft-nav-btn {
    display: block;
    padding: 7px 10px;
    margin-bottom: 4px;
    color: #8b949e;
    text-decoration: none;
    font-size: 13px;
    border-radius: 5px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: color 0.15s, background 0.15s, border-color 0.15s;
  }
  .ft-nav-btn:hover {
    color: #e6edf3;
    background: #161b22;
    border-color: #30363d;
    text-decoration: none;
  }

  /* Keyboard shortcut rows */
  .sc-row {
    display: flex;
    align-items: center;
    gap: 4px;
    margin-bottom: 10px;
  }
  .kbd {
    display: inline-block;
    background: #161b22;
    border: 1px solid #30363d;
    border-bottom: 2px solid #21262d;
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 10px;
    color: #58a6ff;
    font-family: 'SF Mono', 'Consolas', monospace;
    line-height: 1.6;
    letter-spacing: 0.04em;
  }
  .sc-label {
    margin-left: 6px;
    font-size: 12px;
    color: #6e7681;
  }

  /* Data & Legal column */
  .ft-data {
    display: block;
    font-size: 12px;
    color: #6e7681;
    margin-bottom: 8px;
    line-height: 1.5;
  }
  .ft-disc {
    display: block;
    color: #484f58;
    font-size: 11px;
    font-style: italic;
    line-height: 1.7;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #161b22;
  }

  /* Copyright strip */
  .ft-bottom {
    background: #000000;
    border-top: 1px solid #161b22;
    padding: 10px 48px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 11px;
    color: #484f58;
    font-family: 'SF Mono', 'Consolas', monospace;
    letter-spacing: 0.02em;
  }
  .sep { color: #21262d; }
</style>
</head>
<body>

<div class="ft">
  <div class="ft-inner">

    <!-- Brand -->
    <div>
      <span class="ft-logo">&#127974;&nbsp; AI FINANCIAL ADVISOR</span>
      <span class="ft-tagline">Intelligent investment analytics powered by artificial intelligence.</span>
      <span class="ft-sub">Professional-grade tools for portfolio optimization, equity analysis,
      and document intelligence &mdash; built for students and practitioners.</span>
    </div>

    <!-- Navigation -->
    <div>
      <span class="ft-hdr">Navigation</span>
      <a class="ft-nav-btn" data-nav="home">&#127968;&nbsp;&nbsp;Home</a>
      <a class="ft-nav-btn" data-nav="portfolio">&#128202;&nbsp;&nbsp;Portfolio Allocator</a>
      <a class="ft-nav-btn" data-nav="analyzer">&#128200;&nbsp;&nbsp;Stock Analyzer</a>
      <a class="ft-nav-btn" data-nav="caseqa">&#128218;&nbsp;&nbsp;Case Q&amp;A</a>
    </div>

    <!-- Keyboard Shortcuts -->
    <div>
      <span class="ft-hdr">Keyboard Shortcuts</span>
      <div class="sc-row"><span class="kbd">Alt</span><span class="kbd">H</span><span class="sc-label">Home</span></div>
      <div class="sc-row"><span class="kbd">Alt</span><span class="kbd">P</span><span class="sc-label">Portfolio Allocator</span></div>
      <div class="sc-row"><span class="kbd">Alt</span><span class="kbd">S</span><span class="sc-label">Stock Analyzer</span></div>
      <div class="sc-row"><span class="kbd">Alt</span><span class="kbd">C</span><span class="sc-label">Case Q&amp;A</span></div>
    </div>

    <!-- Data & Legal -->
    <div>
      <span class="ft-hdr">Data &amp; Legal</span>
      <span class="ft-data">&#128225;&nbsp; Market data via Yahoo Finance</span>
      <span class="ft-data">&#8987;&nbsp; Prices delayed 15&ndash;20 minutes</span>
      <span class="ft-data">&#129302;&nbsp; AI analysis via OpenAI GPT-4o</span>
      <span class="ft-data">&#127760;&nbsp; Coverage: US-listed equities</span>
      <span class="ft-disc">This platform is for educational purposes only and does not
      constitute financial, investment, or legal advice. Past performance is not
      indicative of future results.</span>
    </div>

  </div>

  <!-- Copyright strip -->
  <div class="ft-bottom">
    <span>&copy; 2025 AI Financial Advisor</span>
    <span class="sep">&middot;</span>
    <span>Not affiliated with any financial institution</span>
    <span class="sep">&middot;</span>
    <span>For educational use only</span>
    <span class="sep">&middot;</span>
    <span>All rights reserved</span>
  </div>
</div>

<script>
(function () {
  function goTo(view) {
    try {
      var url = new URL(window.parent.location.href);
      url.searchParams.set('nav', view);
      window.parent.location.href = url.toString();
    } catch (e) {}
  }

  // Footer nav buttons (inside this iframe — no parent DOM query needed)
  document.querySelectorAll('.ft-nav-btn[data-nav]').forEach(function (el) {
    el.addEventListener('click', function (e) {
      e.preventDefault();
      goTo(el.getAttribute('data-nav'));
    });
  });

  // Keyboard shortcuts (listen on parent window)
  try {
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
  } catch (e) {}
})();
</script>

</body>
</html>
""", height=290, scrolling=False)
