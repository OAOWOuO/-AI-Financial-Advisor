"""
Stock Analyzer - Technical & Fundamental Analysis
Deep-dive analysis of individual stocks with AI recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from typing import Dict, Optional

# ============== PAGE CONFIG ==============
st.set_page_config(
    page_title="Stock Analyzer | AI Financial Advisor",
    page_icon="📈",
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

    /* Text styling */
    h1, h2, h3, h4 { color: #e6edf3 !important; font-weight: 600 !important; }
    p, span, label, li, div { color: #c9d1d9 !important; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background: #161b22; padding: 8px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #8b949e; border-radius: 6px; padding: 8px 16px; }
    .stTabs [aria-selected="true"] { background: #238636 !important; color: white !important; }

    /* Input styling */
    .stTextInput > div > div > input {
        background: #161b22 !important;
        color: #e6edf3 !important;
        border: 1px solid #30363d !important;
    }
    .stSelectbox > div > div {
        background: #161b22 !important;
    }

    /* Metric cards */
    [data-testid="stMetricValue"] { font-size: 24px !important; }

    /* Button styling */
    .stButton > button {
        background: #238636 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
    }
    .stButton > button:hover {
        background: #2ea043 !important;
    }

    /* Recommendation box */
    .rec-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
    }
    .rec-buy { border-left: 4px solid #3fb950; }
    .rec-hold { border-left: 4px solid #d29922; }
    .rec-sell { border-left: 4px solid #f85149; }
</style>
""", unsafe_allow_html=True)


# ============== DATA FUNCTIONS ==============
@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str) -> Dict:
    """Fetch comprehensive stock data."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1y")

        if hist.empty:
            return {"valid": False, "ticker": ticker}

        return {
            "valid": True,
            "ticker": ticker,
            "info": info,
            "history": hist,
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice") or float(hist['Close'].iloc[-1]),
            "change": info.get("regularMarketChange", 0),
            "change_pct": info.get("regularMarketChangePercent", 0),
            "market_cap": info.get("marketCap", 0),
            "volume": info.get("volume", 0),
            "avg_volume": info.get("averageVolume", 0),
            "high_52w": info.get("fiftyTwoWeekHigh", 0),
            "low_52w": info.get("fiftyTwoWeekLow", 0),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "peg_ratio": info.get("pegRatio"),
            "dividend_yield": info.get("dividendYield", 0),
            "beta": info.get("beta", 1),
            "eps": info.get("trailingEps"),
            "revenue": info.get("totalRevenue"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "profit_margin": info.get("profitMargins"),
            "operating_margin": info.get("operatingMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "free_cash_flow": info.get("freeCashflow"),
            "target_price": info.get("targetMeanPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "analyst_rating": info.get("recommendationKey"),
            "num_analysts": info.get("numberOfAnalystOpinions", 0),
        }
    except Exception as e:
        return {"valid": False, "ticker": ticker, "error": str(e)}


def calculate_technical_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators."""
    df = hist.copy()

    # Moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

    # Volume MA
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()

    return df


def generate_recommendation(data: Dict, tech_df: pd.DataFrame) -> Dict:
    """Generate AI recommendation based on technical and fundamental analysis."""
    score = 0
    signals = []
    max_score = 0

    # Technical Signals
    if len(tech_df) > 50:
        latest = tech_df.iloc[-1]
        prev = tech_df.iloc[-2]

        # RSI
        max_score += 2
        if latest['RSI'] < 30:
            score += 2
            signals.append(("RSI Oversold", "bullish", f"RSI at {latest['RSI']:.0f} - potential bounce"))
        elif latest['RSI'] > 70:
            score -= 2
            signals.append(("RSI Overbought", "bearish", f"RSI at {latest['RSI']:.0f} - potential pullback"))
        elif 40 <= latest['RSI'] <= 60:
            score += 1
            signals.append(("RSI Neutral", "neutral", f"RSI at {latest['RSI']:.0f} - balanced"))

        # MACD
        max_score += 2
        if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']:
            score += 2
            signals.append(("MACD Bullish Cross", "bullish", "MACD crossed above signal line"))
        elif latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']:
            score -= 2
            signals.append(("MACD Bearish Cross", "bearish", "MACD crossed below signal line"))
        elif latest['MACD'] > latest['MACD_Signal']:
            score += 1
            signals.append(("MACD Bullish", "bullish", "MACD above signal line"))
        else:
            score -= 1
            signals.append(("MACD Bearish", "bearish", "MACD below signal line"))

        # Moving Averages
        max_score += 2
        price = latest['Close']
        if pd.notna(latest['MA50']) and pd.notna(latest['MA200']):
            if latest['MA50'] > latest['MA200']:
                score += 1
                signals.append(("Golden Cross", "bullish", "50-day MA above 200-day MA"))
            else:
                score -= 1
                signals.append(("Death Cross", "bearish", "50-day MA below 200-day MA"))

            if price > latest['MA50']:
                score += 1
                signals.append(("Above MA50", "bullish", "Price above 50-day moving average"))
            else:
                score -= 1
                signals.append(("Below MA50", "bearish", "Price below 50-day moving average"))

        # Bollinger Bands
        max_score += 1
        if pd.notna(latest['BB_Lower']) and price < latest['BB_Lower']:
            score += 1
            signals.append(("Below BB Lower", "bullish", "Price below lower Bollinger Band - oversold"))
        elif pd.notna(latest['BB_Upper']) and price > latest['BB_Upper']:
            score -= 1
            signals.append(("Above BB Upper", "bearish", "Price above upper Bollinger Band - overbought"))

    # Fundamental Signals
    pe = data.get('pe_ratio')
    if pe:
        max_score += 2
        if pe < 15:
            score += 2
            signals.append(("Low P/E", "bullish", f"P/E of {pe:.1f} suggests undervaluation"))
        elif pe < 25:
            score += 1
            signals.append(("Fair P/E", "neutral", f"P/E of {pe:.1f} is reasonable"))
        elif pe > 40:
            score -= 2
            signals.append(("High P/E", "bearish", f"P/E of {pe:.1f} suggests overvaluation"))
        else:
            score -= 1
            signals.append(("Elevated P/E", "bearish", f"P/E of {pe:.1f} is above average"))

    # Revenue Growth
    rev_growth = data.get('revenue_growth')
    if rev_growth:
        max_score += 2
        if rev_growth > 0.2:
            score += 2
            signals.append(("Strong Growth", "bullish", f"Revenue growth of {rev_growth*100:.0f}%"))
        elif rev_growth > 0.05:
            score += 1
            signals.append(("Moderate Growth", "neutral", f"Revenue growth of {rev_growth*100:.0f}%"))
        elif rev_growth < 0:
            score -= 2
            signals.append(("Declining Revenue", "bearish", f"Revenue declining {rev_growth*100:.0f}%"))

    # Analyst Rating
    rating = data.get('analyst_rating')
    if rating:
        max_score += 2
        if rating in ['strongBuy', 'buy']:
            score += 2
            signals.append(("Analyst Buy", "bullish", f"Analyst consensus: {rating}"))
        elif rating == 'hold':
            score += 0
            signals.append(("Analyst Hold", "neutral", "Analyst consensus: Hold"))
        else:
            score -= 2
            signals.append(("Analyst Sell", "bearish", f"Analyst consensus: {rating}"))

    # Price Target
    target = data.get('target_price')
    price = data.get('price')
    if target and price:
        max_score += 2
        upside = (target - price) / price * 100
        if upside > 20:
            score += 2
            signals.append(("High Upside", "bullish", f"Target ${target:.0f} implies {upside:.0f}% upside"))
        elif upside > 5:
            score += 1
            signals.append(("Moderate Upside", "neutral", f"Target ${target:.0f} implies {upside:.0f}% upside"))
        elif upside < -10:
            score -= 2
            signals.append(("Downside Risk", "bearish", f"Target ${target:.0f} implies {upside:.0f}% downside"))

    # Determine recommendation
    if max_score > 0:
        score_pct = score / max_score * 100
    else:
        score_pct = 0

    if score_pct >= 40:
        recommendation = "BUY"
        rec_class = "rec-buy"
        rec_text = "Technical and fundamental indicators suggest this stock has strong potential."
    elif score_pct >= -20:
        recommendation = "HOLD"
        rec_class = "rec-hold"
        rec_text = "Mixed signals. Consider holding current position or wait for better entry."
    else:
        recommendation = "SELL"
        rec_class = "rec-sell"
        rec_text = "Multiple warning signs. Consider reducing exposure or avoiding new positions."

    return {
        "recommendation": recommendation,
        "rec_class": rec_class,
        "rec_text": rec_text,
        "score": score,
        "max_score": max_score,
        "score_pct": score_pct,
        "signals": signals
    }


# ============== HEADER ==============
col_back, col_title = st.columns([1, 11])
with col_back:
    st.markdown('<a href="/" target="_self" style="color:#58a6ff; text-decoration:none;">← Home</a>', unsafe_allow_html=True)
with col_title:
    st.write("# 📈 Stock Analyzer")
    st.caption("Technical & Fundamental Analysis with AI Recommendations")

# ============== MAIN LAYOUT ==============
col_input, col_analysis = st.columns([1, 3])

# ============== LEFT SIDE - INPUT ==============
with col_input:
    st.subheader("Stock Selection")

    ticker = st.text_input("Enter Ticker", value="AAPL", help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)")
    ticker = ticker.strip().upper()

    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

    # Store in session state
    if 'analyzed_ticker' not in st.session_state:
        st.session_state.analyzed_ticker = None
        st.session_state.stock_data = None

    if analyze_btn and ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            data = fetch_stock_data(ticker)
            if data.get("valid"):
                st.session_state.analyzed_ticker = ticker
                st.session_state.stock_data = data
            else:
                st.error(f"Could not fetch data for {ticker}")

    # Show stock info if analyzed
    if st.session_state.stock_data and st.session_state.stock_data.get("valid"):
        data = st.session_state.stock_data
        st.divider()

        st.write(f"### {data['name']}")
        st.caption(f"{data['sector']} • {data['industry']}")

        # Price
        price = data.get('price', 0)
        change = data.get('change', 0)
        change_pct = data.get('change_pct', 0)

        st.metric(
            "Price",
            f"${price:.2f}",
            delta=f"{change:+.2f} ({change_pct:+.2f}%)"
        )

        # Quick stats
        st.write("**Quick Stats**")
        st.write(f"Market Cap: ${data.get('market_cap', 0)/1e9:.1f}B")
        st.write(f"Volume: {data.get('volume', 0):,.0f}")
        st.write(f"52W Range: ${data.get('low_52w', 0):.0f} - ${data.get('high_52w', 0):.0f}")

        # 52-week position
        high = data.get('high_52w', 0)
        low = data.get('low_52w', 0)
        if high > low:
            pos = (price - low) / (high - low) * 100
            st.progress(pos / 100, text=f"52W Position: {pos:.0f}%")

# ============== RIGHT SIDE - ANALYSIS ==============
with col_analysis:
    if st.session_state.stock_data and st.session_state.stock_data.get("valid"):
        data = st.session_state.stock_data
        hist = data.get("history")

        if hist is not None and not hist.empty:
            # Calculate indicators
            tech_df = calculate_technical_indicators(hist)

            # Generate recommendation
            rec = generate_recommendation(data, tech_df)

            # Show recommendation at top
            st.markdown(f"""
            <div class="rec-box {rec['rec_class']}">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-size: 28px; font-weight: 700; color: {'#3fb950' if rec['recommendation'] == 'BUY' else '#d29922' if rec['recommendation'] == 'HOLD' else '#f85149'};">
                            {rec['recommendation']}
                        </span>
                        <span style="color: #8b949e; margin-left: 15px;">
                            Score: {rec['score']}/{rec['max_score']} ({rec['score_pct']:.0f}%)
                        </span>
                    </div>
                </div>
                <p style="margin-top: 10px; color: #c9d1d9;">{rec['rec_text']}</p>
            </div>
            """, unsafe_allow_html=True)

            # Tabs for analysis
            tab_technical, tab_fundamental = st.tabs(["📊 Technical Analysis", "📋 Fundamental Analysis"])

            # ============== TECHNICAL ANALYSIS TAB ==============
            with tab_technical:
                # Price Chart with MAs
                st.write("### Price & Moving Averages")

                # Prepare data for chart
                chart_data = tech_df.reset_index()
                chart_data['Date'] = pd.to_datetime(chart_data['Date']).dt.tz_localize(None)

                # Create layered chart
                base = alt.Chart(chart_data).encode(
                    x=alt.X('Date:T', title=None, axis=alt.Axis(labelColor='#8b949e', format='%b %Y'))
                )

                price_line = base.mark_line(color='#58a6ff', strokeWidth=2).encode(
                    y=alt.Y('Close:Q', title='Price ($)', axis=alt.Axis(labelColor='#8b949e')),
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('Close:Q', format='$.2f', title='Price')]
                )

                ma50_line = base.mark_line(color='#f0883e', strokeWidth=1, strokeDash=[5,5]).encode(
                    y='MA50:Q',
                    tooltip=[alt.Tooltip('MA50:Q', format='$.2f', title='MA50')]
                )

                ma200_line = base.mark_line(color='#a371f7', strokeWidth=1, strokeDash=[5,5]).encode(
                    y='MA200:Q',
                    tooltip=[alt.Tooltip('MA200:Q', format='$.2f', title='MA200')]
                )

                price_chart = (price_line + ma50_line + ma200_line).properties(
                    height=300
                ).configure_view(strokeWidth=0).configure(background='#161b22')

                st.altair_chart(price_chart, use_container_width=True)
                st.caption("🔵 Price  🟠 50-day MA  🟣 200-day MA")

                # Volume Chart
                st.write("### Volume Analysis")
                vol_chart = alt.Chart(chart_data).mark_bar(color='#58a6ff', opacity=0.6).encode(
                    x=alt.X('Date:T', title=None, axis=alt.Axis(labelColor='#8b949e', format='%b %Y')),
                    y=alt.Y('Volume:Q', title='Volume', axis=alt.Axis(labelColor='#8b949e')),
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('Volume:Q', format=',')]
                ).properties(height=150).configure_view(strokeWidth=0).configure(background='#161b22')

                st.altair_chart(vol_chart, use_container_width=True)

                # RSI and MACD
                col_rsi, col_macd = st.columns(2)

                with col_rsi:
                    st.write("### RSI (14)")
                    latest_rsi = tech_df['RSI'].iloc[-1]

                    rsi_chart = alt.Chart(chart_data.tail(100)).mark_line(color='#58a6ff').encode(
                        x=alt.X('Date:T', title=None, axis=alt.Axis(labelColor='#8b949e')),
                        y=alt.Y('RSI:Q', title='RSI', scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelColor='#8b949e')),
                        tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('RSI:Q', format='.1f')]
                    )

                    # Overbought/oversold lines
                    rules = alt.Chart(pd.DataFrame({'y': [30, 70]})).mark_rule(strokeDash=[3,3], color='#6e7681').encode(y='y:Q')

                    st.altair_chart((rsi_chart + rules).properties(height=200).configure_view(strokeWidth=0).configure(background='#161b22'), use_container_width=True)

                    if latest_rsi > 70:
                        st.warning(f"RSI: {latest_rsi:.0f} - Overbought")
                    elif latest_rsi < 30:
                        st.success(f"RSI: {latest_rsi:.0f} - Oversold")
                    else:
                        st.info(f"RSI: {latest_rsi:.0f} - Neutral")

                with col_macd:
                    st.write("### MACD")

                    macd_line = alt.Chart(chart_data.tail(100)).mark_line(color='#58a6ff').encode(
                        x=alt.X('Date:T', title=None, axis=alt.Axis(labelColor='#8b949e')),
                        y=alt.Y('MACD:Q', title='MACD', axis=alt.Axis(labelColor='#8b949e')),
                        tooltip=[alt.Tooltip('MACD:Q', format='.2f')]
                    )

                    signal_line = alt.Chart(chart_data.tail(100)).mark_line(color='#f0883e').encode(
                        x='Date:T',
                        y='MACD_Signal:Q',
                        tooltip=[alt.Tooltip('MACD_Signal:Q', format='.2f', title='Signal')]
                    )

                    macd_hist = alt.Chart(chart_data.tail(100)).mark_bar(opacity=0.5).encode(
                        x='Date:T',
                        y='MACD_Hist:Q',
                        color=alt.condition(
                            alt.datum.MACD_Hist > 0,
                            alt.value('#3fb950'),
                            alt.value('#f85149')
                        )
                    )

                    st.altair_chart((macd_hist + macd_line + signal_line).properties(height=200).configure_view(strokeWidth=0).configure(background='#161b22'), use_container_width=True)

                    latest_macd = tech_df['MACD'].iloc[-1]
                    latest_signal = tech_df['MACD_Signal'].iloc[-1]
                    if latest_macd > latest_signal:
                        st.success("MACD above signal - Bullish")
                    else:
                        st.warning("MACD below signal - Bearish")

                # Technical Signals Summary
                st.write("### Technical Signals")
                tech_signals = [s for s in rec['signals'] if s[0] in ['RSI Oversold', 'RSI Overbought', 'RSI Neutral',
                                                                       'MACD Bullish Cross', 'MACD Bearish Cross', 'MACD Bullish', 'MACD Bearish',
                                                                       'Golden Cross', 'Death Cross', 'Above MA50', 'Below MA50',
                                                                       'Below BB Lower', 'Above BB Upper']]
                for name, sentiment, desc in tech_signals:
                    if sentiment == 'bullish':
                        st.success(f"**{name}**: {desc}")
                    elif sentiment == 'bearish':
                        st.error(f"**{name}**: {desc}")
                    else:
                        st.info(f"**{name}**: {desc}")

            # ============== FUNDAMENTAL ANALYSIS TAB ==============
            with tab_fundamental:
                # Valuation Metrics
                st.write("### Valuation Metrics")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    pe = data.get('pe_ratio')
                    st.metric("P/E Ratio", f"{pe:.1f}" if pe else "N/A")
                with col2:
                    fpe = data.get('forward_pe')
                    st.metric("Forward P/E", f"{fpe:.1f}" if fpe else "N/A")
                with col3:
                    pb = data.get('pb_ratio')
                    st.metric("P/B Ratio", f"{pb:.1f}" if pb else "N/A")
                with col4:
                    ps = data.get('ps_ratio')
                    st.metric("P/S Ratio", f"{ps:.1f}" if ps else "N/A")

                st.divider()

                # Growth & Profitability
                st.write("### Growth & Profitability")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    rg = data.get('revenue_growth')
                    st.metric("Revenue Growth", f"{rg*100:.1f}%" if rg else "N/A")
                with col2:
                    eg = data.get('earnings_growth')
                    st.metric("Earnings Growth", f"{eg*100:.1f}%" if eg else "N/A")
                with col3:
                    pm = data.get('profit_margin')
                    st.metric("Profit Margin", f"{pm*100:.1f}%" if pm else "N/A")
                with col4:
                    roe = data.get('roe')
                    st.metric("ROE", f"{roe*100:.1f}%" if roe else "N/A")

                st.divider()

                # Financial Health
                st.write("### Financial Health")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    de = data.get('debt_to_equity')
                    st.metric("Debt/Equity", f"{de:.1f}" if de else "N/A")
                with col2:
                    cr = data.get('current_ratio')
                    st.metric("Current Ratio", f"{cr:.1f}" if cr else "N/A")
                with col3:
                    fcf = data.get('free_cash_flow')
                    st.metric("Free Cash Flow", f"${fcf/1e9:.1f}B" if fcf else "N/A")
                with col4:
                    div = data.get('dividend_yield')
                    st.metric("Dividend Yield", f"{div*100:.2f}%" if div else "0%")

                st.divider()

                # Analyst Targets
                st.write("### Analyst Price Targets")
                target = data.get('target_price')
                target_high = data.get('target_high')
                target_low = data.get('target_low')
                price = data.get('price')
                num_analysts = data.get('num_analysts', 0)

                if target and price:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Target Low", f"${target_low:.0f}" if target_low else "N/A")
                    with col2:
                        st.metric("Target Mean", f"${target:.0f}")
                    with col3:
                        st.metric("Target High", f"${target_high:.0f}" if target_high else "N/A")
                    with col4:
                        upside = (target - price) / price * 100
                        st.metric("Implied Upside", f"{upside:+.1f}%")

                    st.caption(f"Based on {num_analysts} analyst estimates")

                    # Visual target range
                    if target_low and target_high:
                        range_data = pd.DataFrame({
                            'Type': ['Current Price', 'Target Mean'],
                            'Price': [price, target]
                        })

                        st.write(f"Current: ${price:.2f} → Target: ${target:.2f} ({upside:+.1f}%)")
                        st.progress(min(1, max(0, (price - target_low) / (target_high - target_low))),
                                   text=f"${target_low:.0f} ← Current ${price:.0f} → ${target_high:.0f}")

                st.divider()

                # Fundamental Signals Summary
                st.write("### Fundamental Signals")
                fund_signals = [s for s in rec['signals'] if s[0] not in ['RSI Oversold', 'RSI Overbought', 'RSI Neutral',
                                                                          'MACD Bullish Cross', 'MACD Bearish Cross', 'MACD Bullish', 'MACD Bearish',
                                                                          'Golden Cross', 'Death Cross', 'Above MA50', 'Below MA50',
                                                                          'Below BB Lower', 'Above BB Upper']]
                for name, sentiment, desc in fund_signals:
                    if sentiment == 'bullish':
                        st.success(f"**{name}**: {desc}")
                    elif sentiment == 'bearish':
                        st.error(f"**{name}**: {desc}")
                    else:
                        st.info(f"**{name}**: {desc}")

    else:
        st.info("👈 Enter a stock ticker and click **Analyze** to see technical and fundamental analysis.")
