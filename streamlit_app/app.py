"""
AI Financial Advisor
Combined app with session state navigation
"""

import streamlit as st
from datetime import datetime
current_year = datetime.now().year

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

/* ══════════════════════════════════════════════════════
   BUTTON SYSTEM — Primary vs Outline variants
   Uses button[kind] attributes that Streamlit renders on every button.
   No :has() needed — these selectors always match.

   Outline  → Enter Portfolio / Enter Analyzer / Enter Case Q&A
   Primary  → Run Full Analysis
   Footer   → text-link override (stHorizontalBlock:has(.ft-nav-section))
   ══════════════════════════════════════════════════════ */

/* ── Base reset for ALL buttons (every selector Streamlit has ever used) ── */
.stButton > button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"],
[data-testid="baseButton-secondary"],
[data-testid="baseButton-primary"] {
    min-height: 44px !important;
    padding: 0 20px !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    box-shadow: none !important;
    text-shadow: none !important;
    transition: background 120ms ease, border-color 120ms ease, transform 60ms ease !important;
    cursor: pointer !important;
}

/* ══ OUTLINE variant — Enter … CTA buttons ══
   Targets every form Streamlit uses to mark a secondary button        */
.stButton > button[kind="secondary"],
button[kind="secondary"],
[data-testid="stBaseButton-secondary"],
[data-testid="baseButton-secondary"] {
    background: transparent !important;
    border: 1.5px solid rgba(255,255,255,0.35) !important;
    color: rgba(255,255,255,0.90) !important;
}
.stButton > button[kind="secondary"]:hover,
button[kind="secondary"]:hover,
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="baseButton-secondary"]:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.55) !important;
    color: #ffffff !important;
}
.stButton > button[kind="secondary"]:active,
button[kind="secondary"]:active,
[data-testid="stBaseButton-secondary"]:active,
[data-testid="baseButton-secondary"]:active {
    transform: translateY(1px) !important;
    box-shadow: none !important;
}
.stButton > button[kind="secondary"]:focus-visible,
button[kind="secondary"]:focus-visible,
[data-testid="stBaseButton-secondary"]:focus-visible,
[data-testid="baseButton-secondary"]:focus-visible {
    outline: 2px solid rgba(120,170,255,0.55) !important;
    outline-offset: 2px !important;
    box-shadow: none !important;
}

/* ══ PRIMARY variant — Run Full Analysis ══
   Visibly different: blue-tinted fill + brighter blue border            */
.stButton > button[kind="primary"],
button[kind="primary"],
[data-testid="stBaseButton-primary"],
[data-testid="baseButton-primary"] {
    background: rgba(99,130,220,0.20) !important;
    border: 1.5px solid rgba(120,170,255,0.65) !important;
    color: rgba(200,220,255,0.95) !important;
}
.stButton > button[kind="primary"]:hover,
button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background: rgba(99,130,220,0.30) !important;
    border-color: rgba(120,170,255,0.85) !important;
    color: #ffffff !important;
    box-shadow: none !important;
}
.stButton > button[kind="primary"]:active,
button[kind="primary"]:active,
[data-testid="stBaseButton-primary"]:active,
[data-testid="baseButton-primary"]:active {
    transform: translateY(1px) !important;
    box-shadow: none !important;
}
.stButton > button[kind="primary"]:focus-visible,
button[kind="primary"]:focus-visible,
[data-testid="stBaseButton-primary"]:focus-visible,
[data-testid="baseButton-primary"]:focus-visible {
    outline: 2px solid rgba(120,170,255,0.80) !important;
    outline-offset: 2px !important;
    box-shadow: none !important;
}

/* ── SEND → (form submit — keep blue tint) ── */
[data-testid="stFormSubmitButton"] > button {
    background: rgba(30,58,110,0.6) !important;
    border: 1px solid rgba(45,90,158,0.7) !important;
    color: #93c5fd !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    height: 36px !important;
    padding: 0 16px !important;
    box-shadow: none !important;
    border-radius: 8px !important;
}
[data-testid="stFormSubmitButton"] > button:hover {
    background: rgba(22,45,84,0.8) !important;
    border-color: rgba(45,90,158,0.9) !important;
    color: #bfdbfe !important;
    box-shadow: none !important;
}
[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(1px) !important;
    box-shadow: none !important;
}

.tool-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.tool-card:hover { border-color: #58a6ff; box-shadow: 0 4px 24px rgba(88,166,255,0.1); }
.tool-card-green  { border-top: 3px solid #10b981 !important; }
.tool-card-indigo { border-top: 3px solid #6366f1 !important; }
.tool-card-purple { border-top: 3px solid #a855f7 !important; }

/* ═══════════════════════════════════════
   FOOTER
   Layout: st.columns() for main content + st.markdown() for copyright.
   Gap between them closed by setting columns padding-bottom: 2rem and
   applying margin-top: -2rem on the copyright element-container.
   Whatever the Streamlit flex gap is (typically 0.5-1rem), the 2rem
   overlap eats into the columns' bottom padding (not the content),
   so the copyright always sits flush with no visible seam.
   ═══════════════════════════════════════ */

/* ─ Footer columns block ─ */
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) {
    margin-top: 48px;
    background: #07111f !important;
    border-top: 3px solid;
    border-image: linear-gradient(90deg, #0a1628 0%, #1d4ed8 42%, #3b82f6 58%, #0a1628 100%) 1;
    gap: 0 !important;
    align-items: stretch !important;
    flex-wrap: wrap;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"] {
    padding: 0 !important;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:not(:last-child) {
    border-right: 1px solid #0d1f38;
}
/* padding-bottom: 2rem (32px) is the key anchor for the gap-close trick below */
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"] > [data-testid="stVerticalBlock"] {
    padding: 28px 28px 32px !important;
    background: #07111f !important;
    gap: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-start !important;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:first-child > [data-testid="stVerticalBlock"] {
    padding-left: 44px !important;
}
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:last-child > [data-testid="stVerticalBlock"] {
    padding-right: 44px !important;
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
/* Reset the global color override so footer text uses its own colors */
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) span,
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) div,
[data-testid="stHorizontalBlock"]:has(.ft-nav-section) p { color: inherit !important; }

/* ─ Copyright strip: margin-top -2rem eats into the columns' 32px bottom padding,
   guaranteeing flush contact regardless of the Streamlit flex gap value ─ */
[data-testid="element-container"]:has(.ft-copy) {
    margin-top: -2rem !important;
    padding: 0 !important;
    line-height: 0 !important;
}
.ft-copy {
    background: #030c17;
    border-top: 1px solid #0d1f38;
    padding: 10px 44px;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 9.5px;
    color: #253d52 !important;
    letter-spacing: 0.05em;
    line-height: 1.4;
}
.ft-copy-sep { color: #142333 !important; }

/* ─ Footer typography ─ */
.ft-section-label {
    display: block;
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #3b82f6 !important;
    margin-bottom: 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #0d1f38;
}
.ft-brand-title {
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #b8cfe8 !important;
    /* Match footer-col-title height exactly so all 4 headers baseline-align */
    height: 1.5rem;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #0d1f38;
    width: 100%;
    margin-bottom: 8px;
    padding-bottom: 8px;
}
.ft-brand-body {
    display: block;
    font-size: 12px;
    line-height: 1.75;
    color: #4e7a9a !important;
}
.ft-data-item {
    font-size: 11.5px;
    color: #4e7a9a !important;
    margin-bottom: 7px;
    line-height: 1.5;
}
.ft-legal-note {
    display: block;
    font-size: 10px;
    color: #2a4d65 !important;
    line-height: 1.65;
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid #0d1f38;
    font-style: italic;
}

/* ─ Footer column layout ─ */
.footer-col {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-items: flex-start;
    gap: 0.75rem;
}
.footer-col-title {
    font-family: 'SF Mono', 'Consolas', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #3b82f6 !important;
    height: 1.5rem;
    display: flex;
    align-items: center;
    border-bottom: 1px solid #0d1f38;
    width: 100%;
    margin-bottom: 8px;
    padding-bottom: 8px;
}

/* ─ Footer nav buttons: plain text-link style ─ */
[data-testid="stVerticalBlock"]:has(.ft-nav-section) .stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    color: #6aa3c8 !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 2px 0 !important;
    min-height: unset !important;
    height: auto !important;
    line-height: 1.6 !important;
    width: auto !important;
    box-shadow: none !important;
    letter-spacing: 0.01em !important;
    margin-bottom: 2px !important;
    transition: color 0.2s ease !important;
}
[data-testid="stVerticalBlock"]:has(.ft-nav-section) .stButton > button:hover {
    background: transparent !important;
    border: none !important;
    color: #e2e8f0 !important;
}
[data-testid="stVerticalBlock"]:has(.ft-nav-section) .stButton > button:focus:not(:active) {
    box-shadow: none !important;
    background: transparent !important;
}
[data-testid="stVerticalBlock"]:has(.ft-nav-section) .stButton > button:active {
    background: transparent !important;
    color: #93c5fd !important;
}

/* ─ Footer about links ─ */
.ft-about-link {
    color: #6aa3c8 !important;
    text-decoration: none;
    font-size: 13px;
    line-height: 1.6;
    display: block;
    transition: color 0.2s;
}
.ft-about-link:hover { color: #e2e8f0 !important; }

/* ═══════════════════════════════════════
   INTERACTIVE ELEMENT SYSTEM (0.1)
   Consistent border / hover / active / focus across all clickable UI.
   Footer buttons are excluded (handled above by .ft-nav-section overrides).
   ═══════════════════════════════════════ */

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #21262d !important;
    gap: 2px !important;
    padding-bottom: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border: 1px solid #30363d !important;
    border-bottom: none !important;
    border-radius: 6px 6px 0 0 !important;
    color: #8b949e !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 18px !important;
    margin-bottom: -2px !important;
    transition: border-color 0.18s ease, color 0.18s ease, background 0.18s ease !important;
    cursor: pointer !important;
}
[data-baseweb="tab"]:hover {
    border-color: #58a6ff !important;
    color: #c9d1d9 !important;
    background: #161b22 !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    background: #161b22 !important;
    border-color: #58a6ff !important;
    border-bottom-color: #161b22 !important;
    color: #58a6ff !important;
    font-weight: 600 !important;
}
[data-baseweb="tab"]:focus-visible {
    outline: 2px solid #58a6ff !important;
    outline-offset: 2px !important;
}
/* Hide the sliding underline highlight — we use border instead */
[data-baseweb="tab-highlight"] { background: transparent !important; height: 0 !important; }

/* ── Text inputs & form inputs ── */
[data-testid="stTextInput"] > div > div,
[data-baseweb="base-input"] {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}
[data-testid="stTextInput"] > div > div:focus-within,
[data-baseweb="base-input"]:focus-within {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}
[data-testid="stTextInput"] input {
    color: #e6edf3 !important;
    background: transparent !important;
}
[data-testid="stTextInput"] input::placeholder { color: #6e7681 !important; }

/* ── Radio buttons (e.g. chart type toggle) ── */
[data-testid="stRadio"] > div {
    display: flex !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
}
[data-testid="stRadio"] label {
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    padding: 5px 14px !important;
    cursor: pointer !important;
    transition: border-color 0.18s ease, color 0.18s ease, background 0.18s ease !important;
    color: #8b949e !important;
}
[data-testid="stRadio"] label:hover {
    border-color: #58a6ff !important;
    color: #c9d1d9 !important;
    background: #161b22 !important;
}
/* Active/selected radio state */
[data-testid="stRadio"] label:has(input:checked) {
    border-color: #58a6ff !important;
    background: rgba(88,166,255,0.08) !important;
    color: #58a6ff !important;
}
[data-testid="stRadio"] label:focus-within {
    outline: 2px solid #58a6ff !important;
    outline-offset: 2px !important;
}
/* Hide the default radio circle only — keep text labels visible */
[data-testid="stRadio"] label > div:first-child { display: none !important; }

/* ── Generic button reset: no glow on focus, clean active ── */
.stButton > button:focus-visible {
    outline: 2px solid #30363d !important;
    outline-offset: 2px !important;
    box-shadow: none !important;
}
.stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow: none !important;
}

/* ── Selectbox / dropdown ── */
[data-testid="stSelectbox"] > div > div {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
    transition: border-color 0.18s ease !important;
}
[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stSelectbox"] > div > div:hover {
    border-color: #58a6ff !important;
}

/* ═══════════════════════════════════════
   RESPONSIVE LAYOUT
   ═══════════════════════════════════════ */

/* Tablet (≤ 900 px): footer 2 × 2 */
@media (max-width: 900px) {
    [data-testid="stHorizontalBlock"]:has(.ft-nav-section) {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"] {
        flex: 0 0 50% !important;
        min-width: 50% !important;
        max-width: 50% !important;
        border-right: none !important;
    }
    /* Keep divider only between col 1 and col 2 on the same row */
    [data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:nth-child(1),
    [data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:nth-child(3) {
        border-right: 1px solid #0d1f38 !important;
    }
}

/* Mobile (≤ 560 px): footer single column */
@media (max-width: 560px) {
    [data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"] {
        flex: 0 0 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        border-right: none !important;
    }
    [data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:not(:last-child) {
        border-bottom: 1px solid #0d1f38 !important;
    }
    [data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:first-child > [data-testid="stVerticalBlock"],
    [data-testid="stHorizontalBlock"]:has(.ft-nav-section) > [data-testid="column"]:last-child > [data-testid="stVerticalBlock"] {
        padding-left: 24px !important;
        padding-right: 24px !important;
    }
}

/* Mobile: home tool cards stack vertically */
@media (max-width: 768px) {
    [data-testid="stHorizontalBlock"]:has(.tool-card-green) {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"]:has(.tool-card-green) > [data-testid="column"] {
        flex: 0 0 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
    }
}

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
        st.markdown("""<div class="tool-card tool-card-green">
<div style="font-size: 48px; margin-bottom: 15px;">📊</div>
<div style="font-size: 22px; font-weight: 700; color: #e6edf3 !important; margin-bottom: 4px;">Portfolio Allocator</div>
<div style="font-size: 11px; color: #10b981 !important; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 12px;">RISK &amp; ALLOCATION</div>
<p style="color: #8b949e; margin: 12px 0;">Optimize your multi-stock portfolio with AI-powered allocation, risk analytics, and rebalancing recommendations.</p>
<ul style="text-align: left; color: #8b949e; padding-left: 20px; font-size: 14px; line-height: 1.9;">
<li>Multi-stock signal analysis</li>
<li>Position sizing &amp; allocation</li>
<li>Risk metrics (Sharpe, Beta, VaR)</li>
<li>Performance vs S&amp;P 500</li>
<li>Dividend income tracking</li>
<li>One-click rebalancing</li>
</ul>
</div>""", unsafe_allow_html=True)
        st.markdown('<div class="btn-cta-portfolio"></div>', unsafe_allow_html=True)
        if st.button("Enter Portfolio Allocator →", key="btn_portfolio", use_container_width=True):
            st.session_state.current_view = "portfolio"
            st.rerun()

    with col2:
        st.markdown("""<div class="tool-card tool-card-indigo">
<div style="font-size: 48px; margin-bottom: 15px;">📈</div>
<div style="font-size: 22px; font-weight: 700; color: #e6edf3 !important; margin-bottom: 4px;">Stock Analyzer</div>
<div style="font-size: 11px; color: #818cf8 !important; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 12px;">TECHNICAL &amp; FUNDAMENTAL</div>
<p style="color: #8b949e; margin: 12px 0;">Deep-dive analysis of individual stocks with institutional-grade technical and fundamental scoring to find the best opportunities.</p>
<ul style="text-align: left; color: #8b949e; padding-left: 20px; font-size: 14px; line-height: 1.9;">
<li>Technical analysis (RSI, MACD, Bollinger)</li>
<li>Fundamental scoring (valuation, profitability, growth)</li>
<li>Multi-model valuation (P/E, DCF, consensus)</li>
<li>Return forecasts with confidence intervals</li>
<li>Support &amp; resistance levels</li>
<li>BUY / HOLD / SELL recommendation</li>
</ul>
</div>""", unsafe_allow_html=True)
        st.markdown('<div class="btn-cta-analyzer"></div>', unsafe_allow_html=True)
        if st.button("Enter Stock Analyzer →", key="btn_analyzer", use_container_width=True):
            st.session_state.current_view = "analyzer"
            st.rerun()

    with col3:
        st.markdown("""<div class="tool-card tool-card-purple">
<div style="font-size: 48px; margin-bottom: 15px;">📚</div>
<div style="font-size: 22px; font-weight: 700; color: #e6edf3 !important; margin-bottom: 4px;">Case Q&amp;A</div>
<div style="font-size: 11px; color: #c084fc !important; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 12px;">RAG-POWERED LEARNING</div>
<p style="color: #8b949e; margin: 12px 0;">Chat with your course materials using RAG-powered AI. Every answer is grounded in your uploaded documents with file and page citations.</p>
<ul style="text-align: left; color: #8b949e; padding-left: 20px; font-size: 14px; line-height: 1.9;">
<li>Upload PDFs directly in the browser</li>
<li>Auto-indexed — no terminal needed</li>
<li>Ask any question about the materials</li>
<li>Cited answers: file + page + chunk ID</li>
<li>Refuses unsupported questions explicitly</li>
</ul>
</div>""", unsafe_allow_html=True)
        st.markdown('<div class="btn-cta-caseqa"></div>', unsafe_allow_html=True)
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
        st.markdown('<div class="btn-back"></div>', unsafe_allow_html=True)
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
ft_col1, ft_col2, ft_col3, ft_col4 = st.columns([2, 1, 1.5, 1.5])

with ft_col1:
    st.markdown("""
<span class="ft-brand-title">🏦 AI FINANCIAL ADVISOR</span>
<span class="ft-brand-body">
  Institutional-grade analytics for portfolio optimization,
  equity research, and document intelligence.
</span>
""", unsafe_allow_html=True)

with ft_col2:
    st.markdown('<span class="footer-col-title">Navigation</span><div class="ft-nav-section"></div>', unsafe_allow_html=True)
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
    st.markdown("""
<span class="footer-col-title">Data &amp; Legal</span>
<div class="ft-data-item">• Market data: Yahoo Finance (15–20 min delayed)</div>
<div class="ft-data-item">• AI analysis: OpenAI API</div>
<div class="ft-data-item">• Equity universe: US-listed stocks only</div>
<div class="ft-data-item">• Data accuracy not guaranteed</div>
<span class="ft-legal-note">Past performance is not indicative of future results.
Not affiliated with any financial institution.<br><br>
Source code: <a href="https://github.com/OAOWOuO/-AI-Financial-Advisor" target="_blank"
style="color:#3b82f6 !important;text-decoration:none;">github.com/OAOWOuO/-AI-Financial-Advisor</a></span>
""", unsafe_allow_html=True)

with ft_col4:
    st.markdown("""
<span class="footer-col-title">About</span>
<div class="ft-data-item"><strong style="color:#8b9db5 !important;">Author</strong></div>
<div class="ft-data-item">YuanTeng Fan</div>
<a class="ft-about-link" href="https://www.linkedin.com/in/yuan-teng-fan1208/" target="_blank">LinkedIn →</a>
<div class="ft-data-item" style="margin-top:10px;"><strong style="color:#8b9db5 !important;">Course</strong></div>
<div class="ft-data-item">MGMT 690 · Mastering AI for Finance</div>
<a class="ft-about-link" href="https://business.purdue.edu/" target="_blank">Purdue Business School →</a>
""", unsafe_allow_html=True)

# Copyright strip — flush below the columns via negative margin
st.markdown(f"""
<div class="ft-copy">
  <span>© {current_year} AI Financial Advisor</span>
  <span class="ft-copy-sep">·</span>
  <span>Educational use only</span>
  <span class="ft-copy-sep">·</span>
  <span>Not financial advice</span>
  <span class="ft-copy-sep">·</span>
  <span>Built for MGMT 690 · Purdue University</span>
</div>
""", unsafe_allow_html=True)
