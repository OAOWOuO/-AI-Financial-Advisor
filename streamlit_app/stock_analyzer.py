import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

try:
    openai_api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
except Exception:
    openai_api_key = os.environ.get("OPENAI_API_KEY", "")


# ============== DATA FETCHING ==============
@st.cache_data(ttl=300)
def fetch_comprehensive_data(ticker: str) -> Dict:
    """Fetch all data needed for institutional analysis."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info

        # Price history - multiple timeframes
        hist_1y = stock.history(period="1y")
        hist_2y = stock.history(period="2y")
        hist_5y = stock.history(period="5y")

        if hist_1y.empty:
            return {"valid": False, "ticker": ticker}

        # Financial statements
        income_stmt = stock.income_stmt
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow

        # Quarterly data
        quarterly_income = stock.quarterly_income_stmt
        quarterly_bs = stock.quarterly_balance_sheet

        return {
            "valid": True,
            "ticker": ticker,
            "info": info,
            "hist_1y": hist_1y,
            "hist_2y": hist_2y,
            "hist_5y": hist_5y,
            "income_stmt": income_stmt,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow,
            "quarterly_income": quarterly_income,
            "quarterly_bs": quarterly_bs,
            # Basic info
            "name": info.get("shortName", ticker),
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "price": info.get("currentPrice") or info.get("regularMarketPrice") or float(hist_1y['Close'].iloc[-1]),
            "prev_close": info.get("previousClose", 0),
            "open": info.get("open", 0),
            "high": info.get("dayHigh", 0),
            "low": info.get("dayLow", 0),
            "volume": info.get("volume", 0),
            "avg_volume": info.get("averageVolume", 0),
            "avg_volume_10d": info.get("averageVolume10days", 0),
            "market_cap": info.get("marketCap", 0),
            "enterprise_value": info.get("enterpriseValue", 0),
            "shares_outstanding": info.get("sharesOutstanding", 0),
            "float_shares": info.get("floatShares", 0),
            # 52-week range
            "high_52w": info.get("fiftyTwoWeekHigh", 0),
            "low_52w": info.get("fiftyTwoWeekLow", 0),
            # Valuation metrics
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            "ev_ebitda": info.get("enterpriseToEbitda"),
            "ev_revenue": info.get("enterpriseToRevenue"),
            # Profitability
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            # Growth
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "earnings_quarterly_growth": info.get("earningsQuarterlyGrowth"),
            # Financial health
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "total_debt": info.get("totalDebt"),
            "total_cash": info.get("totalCash"),
            "free_cash_flow": info.get("freeCashflow"),
            "operating_cash_flow": info.get("operatingCashflow"),
            # Per share
            "eps": info.get("trailingEps"),
            "forward_eps": info.get("forwardEps"),
            "book_value": info.get("bookValue"),
            "revenue_per_share": info.get("revenuePerShare"),
            # Dividends
            "dividend_yield": info.get("dividendYield", 0),
            "dividend_rate": info.get("dividendRate", 0),
            "payout_ratio": info.get("payoutRatio"),
            # Analyst data
            "target_price": info.get("targetMeanPrice"),
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "analyst_rating": info.get("recommendationKey"),
            "num_analysts": info.get("numberOfAnalystOpinions", 0),
            # Beta
            "beta": info.get("beta", 1),
        }
    except Exception as e:
        return {"valid": False, "ticker": ticker, "error": str(e)}


@st.cache_data(ttl=600)
def fetch_peer_comparison(ticker: str, sector: str) -> pd.DataFrame:
    """Fetch key metrics for sector peers to enable relative valuation."""
    _SECTOR_PEERS = {
        "Technology": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
        "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T"],
        "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE"],
        "Financial Services": ["JPM", "BAC", "GS", "MS", "BRK-B"],
        "Healthcare": ["JNJ", "UNH", "LLY", "ABBV", "MRK"],
        "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
        "Industrials": ["GE", "HON", "UPS", "CAT", "DE"],
        "Consumer Defensive": ["WMT", "COST", "PG", "KO", "PEP"],
        "Basic Materials": ["LIN", "APD", "ECL", "NEM", "FCX"],
        "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
        "Utilities": ["NEE", "DUK", "SO", "D", "AEP"],
    }
    peers = _SECTOR_PEERS.get(sector, ["SPY", "QQQ", "DIA", "IWM", "VTI"])
    peers = [ticker] + [p for p in peers if p != ticker][:4]

    rows = []
    try:
        import yfinance as yf
        for sym in peers:
            try:
                info = yf.Ticker(sym).info
                pe = info.get("trailingPE")
                ev_ebitda = info.get("enterpriseToEbitda")
                rev_growth = info.get("revenueGrowth")
                net_margin = info.get("profitMargins")
                roe = info.get("returnOnEquity")
                rows.append({
                    "Ticker": sym,
                    "P/E": f"{pe:.1f}x" if pe else "N/A",
                    "EV/EBITDA": f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A",
                    "Rev Growth": f"{rev_growth*100:.1f}%" if rev_growth else "N/A",
                    "Net Margin": f"{net_margin*100:.1f}%" if net_margin else "N/A",
                    "ROE": f"{roe*100:.1f}%" if roe else "N/A",
                    "_is_subject": sym == ticker,
                })
            except Exception:
                pass
    except Exception:
        pass
    return pd.DataFrame(rows)


# ============== TECHNICAL ANALYSIS ==============
def calculate_technical_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive technical indicators."""
    df = hist.copy()

    # Moving Averages
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_100'] = df['Close'].rolling(100).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    df['ATR_Percent'] = df['ATR'] / df['Close'] * 100

    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']

    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # Stochastic
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ADX (Average Directional Index)
    plus_dm = df['High'].diff()
    minus_dm = df['Low'].diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr14 = true_range.rolling(14).sum()
    plus_di = 100 * (plus_dm.rolling(14).sum() / tr14)
    minus_di = 100 * (abs(minus_dm).rolling(14).sum() / tr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    df['ADX'] = dx.rolling(14).mean()
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di

    return df


def identify_support_resistance(hist: pd.DataFrame, lookback: int = 60) -> Tuple[List[float], List[float]]:
    """Identify key support and resistance levels."""
    df = hist.tail(lookback).copy()

    # Find local minima and maxima
    window = 5
    df['Local_Min'] = df['Low'].rolling(window, center=True).min()
    df['Local_Max'] = df['High'].rolling(window, center=True).max()

    supports = df[df['Low'] == df['Local_Min']]['Low'].tolist()
    resistances = df[df['High'] == df['Local_Max']]['High'].tolist()

    # Cluster nearby levels
    def cluster_levels(levels, threshold_pct=0.02):
        if not levels:
            return []
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        for level in levels[1:]:
            if (level - current_cluster[0]) / current_cluster[0] < threshold_pct:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        clustered.append(np.mean(current_cluster))
        return clustered

    supports = cluster_levels(supports)[-3:] if supports else []
    resistances = cluster_levels(resistances)[-3:] if resistances else []

    return supports, resistances


def generate_technical_signals(data: Dict, tech_df: pd.DataFrame) -> Dict:
    """Generate technical signals with explicit thresholds and weights."""
    signals = []
    total_score = 0
    max_score = 0

    latest = tech_df.iloc[-1]
    prev = tech_df.iloc[-2] if len(tech_df) > 1 else latest
    price = data['price']

    # ===== TREND SIGNALS (Weight: 30%) =====
    trend_score = 0
    trend_max = 30

    # MA Alignment (10 points)
    ma_score = 0
    if pd.notna(latest['SMA_20']) and pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']):
        if latest['SMA_20'] > latest['SMA_50'] > latest['SMA_200']:
            ma_score = 10
            signals.append({
                "category": "Trend", "indicator": "MA Alignment",
                "signal": "BULLISH", "score": "+10",
                "detail": "SMA20 > SMA50 > SMA200 - Strong uptrend structure",
                "threshold": "All MAs aligned ascending"
            })
        elif latest['SMA_20'] < latest['SMA_50'] < latest['SMA_200']:
            ma_score = -10
            signals.append({
                "category": "Trend", "indicator": "MA Alignment",
                "signal": "BEARISH", "score": "-10",
                "detail": "SMA20 < SMA50 < SMA200 - Strong downtrend structure",
                "threshold": "All MAs aligned descending"
            })
        else:
            ma_score = 0
            signals.append({
                "category": "Trend", "indicator": "MA Alignment",
                "signal": "NEUTRAL", "score": "0",
                "detail": "Mixed MA alignment - No clear trend",
                "threshold": "MAs not aligned"
            })
    trend_score += ma_score

    # Price vs MAs (10 points)
    price_ma_score = 0
    if pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']):
        above_50 = price > latest['SMA_50']
        above_200 = price > latest['SMA_200']
        if above_50 and above_200:
            price_ma_score = 10
            signals.append({
                "category": "Trend", "indicator": "Price Position",
                "signal": "BULLISH", "score": "+10",
                "detail": f"Price ${price:.2f} above both SMA50 (${latest['SMA_50']:.2f}) and SMA200 (${latest['SMA_200']:.2f})",
                "threshold": "Price > SMA50 AND Price > SMA200"
            })
        elif not above_50 and not above_200:
            price_ma_score = -10
            signals.append({
                "category": "Trend", "indicator": "Price Position",
                "signal": "BEARISH", "score": "-10",
                "detail": f"Price ${price:.2f} below both SMA50 (${latest['SMA_50']:.2f}) and SMA200 (${latest['SMA_200']:.2f})",
                "threshold": "Price < SMA50 AND Price < SMA200"
            })
        else:
            price_ma_score = 0
            signals.append({
                "category": "Trend", "indicator": "Price Position",
                "signal": "NEUTRAL", "score": "0",
                "detail": f"Price between key MAs",
                "threshold": "Mixed position relative to MAs"
            })
    trend_score += price_ma_score

    # Golden/Death Cross (10 points)
    cross_score = 0
    if pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']):
        if latest['SMA_50'] > latest['SMA_200'] and prev['SMA_50'] <= prev['SMA_200']:
            cross_score = 10
            signals.append({
                "category": "Trend", "indicator": "MA Cross",
                "signal": "BULLISH", "score": "+10",
                "detail": "Golden Cross - SMA50 crossed above SMA200",
                "threshold": "SMA50 crosses above SMA200"
            })
        elif latest['SMA_50'] < latest['SMA_200'] and prev['SMA_50'] >= prev['SMA_200']:
            cross_score = -10
            signals.append({
                "category": "Trend", "indicator": "MA Cross",
                "signal": "BEARISH", "score": "-10",
                "detail": "Death Cross - SMA50 crossed below SMA200",
                "threshold": "SMA50 crosses below SMA200"
            })
        elif latest['SMA_50'] > latest['SMA_200']:
            cross_score = 5
            signals.append({
                "category": "Trend", "indicator": "MA Cross",
                "signal": "BULLISH", "score": "+5",
                "detail": "SMA50 above SMA200 (post Golden Cross)",
                "threshold": "SMA50 > SMA200"
            })
        else:
            cross_score = -5
            signals.append({
                "category": "Trend", "indicator": "MA Cross",
                "signal": "BEARISH", "score": "-5",
                "detail": "SMA50 below SMA200 (post Death Cross)",
                "threshold": "SMA50 < SMA200"
            })
    trend_score += cross_score

    total_score += trend_score
    max_score += trend_max

    # ===== MOMENTUM SIGNALS (Weight: 30%) =====
    momentum_score = 0
    momentum_max = 30

    # RSI (15 points)
    rsi = latest['RSI']
    rsi_score = 0
    if pd.notna(rsi):
        if rsi < 30:
            rsi_score = 15
            signals.append({
                "category": "Momentum", "indicator": "RSI",
                "signal": "BULLISH", "score": "+15",
                "detail": f"RSI at {rsi:.1f} - Oversold, potential bounce",
                "threshold": "RSI < 30 (Oversold)"
            })
        elif rsi > 70:
            rsi_score = -15
            signals.append({
                "category": "Momentum", "indicator": "RSI",
                "signal": "BEARISH", "score": "-15",
                "detail": f"RSI at {rsi:.1f} - Overbought, potential pullback",
                "threshold": "RSI > 70 (Overbought)"
            })
        elif 40 <= rsi <= 60:
            rsi_score = 0
            signals.append({
                "category": "Momentum", "indicator": "RSI",
                "signal": "NEUTRAL", "score": "0",
                "detail": f"RSI at {rsi:.1f} - Neutral zone, no directional bias",
                "threshold": "40 ≤ RSI ≤ 60 (Neutral)"
            })
        elif rsi < 40:
            rsi_score = 8
            signals.append({
                "category": "Momentum", "indicator": "RSI",
                "signal": "BULLISH", "score": "+8",
                "detail": f"RSI at {rsi:.1f} - Approaching oversold",
                "threshold": "30 ≤ RSI < 40"
            })
        else:
            rsi_score = -8
            signals.append({
                "category": "Momentum", "indicator": "RSI",
                "signal": "BEARISH", "score": "-8",
                "detail": f"RSI at {rsi:.1f} - Approaching overbought",
                "threshold": "60 < RSI ≤ 70"
            })
    momentum_score += rsi_score

    # MACD (15 points)
    macd_score = 0
    if pd.notna(latest['MACD']) and pd.notna(latest['MACD_Signal']):
        macd_cross_bull = latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal']
        macd_cross_bear = latest['MACD'] < latest['MACD_Signal'] and prev['MACD'] >= prev['MACD_Signal']

        if macd_cross_bull:
            macd_score = 15
            signals.append({
                "category": "Momentum", "indicator": "MACD",
                "signal": "BULLISH", "score": "+15",
                "detail": "MACD bullish crossover - Buy signal",
                "threshold": "MACD crosses above Signal line"
            })
        elif macd_cross_bear:
            macd_score = -15
            signals.append({
                "category": "Momentum", "indicator": "MACD",
                "signal": "BEARISH", "score": "-15",
                "detail": "MACD bearish crossover - Sell signal",
                "threshold": "MACD crosses below Signal line"
            })
        elif latest['MACD'] > latest['MACD_Signal'] and latest['MACD_Hist'] > prev['MACD_Hist']:
            macd_score = 10
            signals.append({
                "category": "Momentum", "indicator": "MACD",
                "signal": "BULLISH", "score": "+10",
                "detail": f"MACD above signal, histogram {latest['MACD_Hist']:+.4f} (expanding from {prev['MACD_Hist']:+.4f})",
                "threshold": "MACD > Signal, Histogram increasing"
            })
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < prev['MACD_Hist']:
            macd_score = -10
            signals.append({
                "category": "Momentum", "indicator": "MACD",
                "signal": "BEARISH", "score": "-10",
                "detail": f"MACD below signal, histogram {latest['MACD_Hist']:+.4f} (expanding from {prev['MACD_Hist']:+.4f})",
                "threshold": "MACD < Signal, Histogram decreasing"
            })
        elif latest['MACD'] > latest['MACD_Signal']:
            macd_score = 5
            signals.append({
                "category": "Momentum", "indicator": "MACD",
                "signal": "BULLISH", "score": "+5",
                "detail": "MACD above signal line",
                "threshold": "MACD > Signal"
            })
        else:
            macd_score = -5
            signals.append({
                "category": "Momentum", "indicator": "MACD",
                "signal": "BEARISH", "score": "-5",
                "detail": "MACD below signal line",
                "threshold": "MACD < Signal"
            })
    momentum_score += macd_score

    total_score += momentum_score
    max_score += momentum_max

    # ===== VOLATILITY SIGNALS (Weight: 20%) =====
    volatility_score = 0
    volatility_max = 20

    # Bollinger Bands (10 points)
    bb_score = 0
    if pd.notna(latest['BB_Position']):
        bb_pos = latest['BB_Position']
        if bb_pos < 0:
            bb_score = 10
            signals.append({
                "category": "Volatility", "indicator": "Bollinger Bands",
                "signal": "BULLISH", "score": "+10",
                "detail": f"Price below lower band - Oversold",
                "threshold": "Price < Lower BB"
            })
        elif bb_pos > 1:
            bb_score = -10
            signals.append({
                "category": "Volatility", "indicator": "Bollinger Bands",
                "signal": "BEARISH", "score": "-10",
                "detail": f"Price above upper band - Overbought",
                "threshold": "Price > Upper BB"
            })
        elif bb_pos < 0.2:
            bb_score = 5
            signals.append({
                "category": "Volatility", "indicator": "Bollinger Bands",
                "signal": "BULLISH", "score": "+5",
                "detail": f"Price near lower band ({bb_pos*100:.0f}%)",
                "threshold": "BB Position < 20%"
            })
        elif bb_pos > 0.8:
            bb_score = -5
            signals.append({
                "category": "Volatility", "indicator": "Bollinger Bands",
                "signal": "BEARISH", "score": "-5",
                "detail": f"Price near upper band ({bb_pos*100:.0f}%)",
                "threshold": "BB Position > 80%"
            })
        else:
            bb_score = 0
            signals.append({
                "category": "Volatility", "indicator": "Bollinger Bands",
                "signal": "NEUTRAL", "score": "0",
                "detail": f"Price in middle of bands ({bb_pos*100:.0f}%)",
                "threshold": "20% ≤ BB Position ≤ 80%"
            })
    volatility_score += bb_score

    # ATR Volatility (10 points)
    atr_score = 0
    if pd.notna(latest['ATR_Percent']):
        atr_pct = latest['ATR_Percent']
        # Compare to historical
        atr_avg = tech_df['ATR_Percent'].tail(60).mean()
        if pd.notna(atr_avg):
            if atr_pct > atr_avg * 1.5:
                atr_score = -5  # High volatility is risky
                signals.append({
                    "category": "Volatility", "indicator": "ATR",
                    "signal": "CAUTION", "score": "-5",
                    "detail": f"ATR {atr_pct:.2f}% - Elevated volatility (avg: {atr_avg:.2f}%)",
                    "threshold": "ATR > 1.5x average"
                })
            elif atr_pct < atr_avg * 0.7:
                atr_score = 5  # Low volatility, potential breakout
                signals.append({
                    "category": "Volatility", "indicator": "ATR",
                    "signal": "WATCH", "score": "+5",
                    "detail": f"ATR {atr_pct:.2f}% - Low volatility, potential breakout setup",
                    "threshold": "ATR < 0.7x average"
                })
            else:
                atr_score = 0
                signals.append({
                    "category": "Volatility", "indicator": "ATR",
                    "signal": "NEUTRAL", "score": "0",
                    "detail": f"ATR {atr_pct:.2f}% - Normal volatility",
                    "threshold": "0.7x < ATR < 1.5x average"
                })
    volatility_score += atr_score

    total_score += volatility_score
    max_score += volatility_max

    # ===== VOLUME SIGNALS (Weight: 20%) =====
    volume_score = 0
    volume_max = 20

    # Volume vs Average (10 points)
    vol_score = 0
    if pd.notna(latest['Volume_Ratio']):
        vol_ratio = latest['Volume_Ratio']
        price_change = (latest['Close'] - prev['Close']) / prev['Close'] * 100 if prev['Close'] > 0 else 0

        if vol_ratio > 1.5 and price_change > 0:
            vol_score = 10
            signals.append({
                "category": "Volume", "indicator": "Volume Surge",
                "signal": "BULLISH", "score": "+10",
                "detail": f"Volume {vol_ratio:.1f}x average with price up {price_change:.1f}%",
                "threshold": "Volume > 1.5x avg + Price up"
            })
        elif vol_ratio > 1.5 and price_change < 0:
            vol_score = -10
            signals.append({
                "category": "Volume", "indicator": "Volume Surge",
                "signal": "BEARISH", "score": "-10",
                "detail": f"Volume {vol_ratio:.1f}x average with price down {price_change:.1f}%",
                "threshold": "Volume > 1.5x avg + Price down"
            })
        elif vol_ratio < 0.5:
            vol_score = -5
            signals.append({
                "category": "Volume", "indicator": "Volume",
                "signal": "CAUTION", "score": "-5",
                "detail": f"Low volume ({vol_ratio:.1f}x avg) - Weak conviction / volume shrinkage risk",
                "threshold": "Volume < 0.5x average"
            })
        else:
            vol_score = 0
            signals.append({
                "category": "Volume", "indicator": "Volume",
                "signal": "NEUTRAL", "score": "0",
                "detail": f"Normal volume ({vol_ratio:.1f}x average)",
                "threshold": "0.5x ≤ Volume ≤ 1.5x average"
            })
    volume_score += vol_score

    # OBV Trend (10 points)
    obv_score = 0
    if len(tech_df) > 20:
        obv_sma = tech_df['OBV'].tail(20).mean()
        obv_current = latest['OBV']
        obv_trend = (obv_current - obv_sma) / abs(obv_sma) * 100 if obv_sma != 0 else 0

        if obv_trend > 10:
            obv_score = 10
            signals.append({
                "category": "Volume", "indicator": "OBV",
                "signal": "BULLISH", "score": "+10",
                "detail": f"OBV trending up {obv_trend:.0f}% above 20-day avg",
                "threshold": "OBV > 10% above average"
            })
        elif obv_trend < -10:
            obv_score = -10
            signals.append({
                "category": "Volume", "indicator": "OBV",
                "signal": "BEARISH", "score": "-10",
                "detail": f"OBV trending down {obv_trend:.0f}% below 20-day avg",
                "threshold": "OBV < -10% below average"
            })
        else:
            obv_score = 0
            _obv_slope_pct = ((latest['OBV'] - tech_df['OBV'].iloc[-10]) / (abs(tech_df['OBV'].iloc[-10]) + 1e-10) * 100) if len(tech_df) > 10 else 0
            _obv_slope_desc = "slight upward slope" if _obv_slope_pct > 2 else "slight downward slope" if _obv_slope_pct < -2 else "flat"
            signals.append({
                "category": "Volume", "indicator": "OBV",
                "signal": "NEUTRAL", "score": "0",
                "detail": f"OBV {_obv_slope_desc} ({obv_trend:+.1f}% vs 20-day avg) — no sustained accumulation or distribution",
                "threshold": "-10% ≤ OBV deviation ≤ 10%"
            })
    volume_score += obv_score

    total_score += volume_score
    max_score += volume_max

    # Calculate percentages
    score_pct = (total_score / max_score * 100) if max_score > 0 else 0

    # Determine technical rating
    if score_pct >= 35:
        tech_rating = ("STRONG BUY", "#00ff00")
    elif score_pct >= 15:
        tech_rating = ("BUY", "#3fb950")
    elif score_pct >= -10:
        tech_rating = ("HOLD", "#d29922")
    elif score_pct >= -30:
        tech_rating = ("SELL", "#f85149")
    else:
        tech_rating = ("STRONG SELL", "#ff0000")

    return {
        "signals": signals,
        "total_score": total_score,
        "max_score": max_score,
        "score_pct": score_pct,
        "rating": tech_rating[0],
        "rating_color": tech_rating[1],
        "breakdown": {
            "trend": {"score": trend_score, "max": trend_max},
            "momentum": {"score": momentum_score, "max": momentum_max},
            "volatility": {"score": volatility_score, "max": volatility_max},
            "volume": {"score": volume_score, "max": volume_max}
        }
    }


# ============== FUNDAMENTAL ANALYSIS ==============
def analyze_fundamentals(data: Dict) -> Dict:
    """Comprehensive fundamental analysis."""
    signals = []
    total_score = 0
    max_score = 0

    # ===== VALUATION (30 points) =====
    valuation_score = 0
    valuation_max = 30

    # P/E Analysis (10 points)
    pe = data.get('pe_ratio')
    forward_pe = data.get('forward_pe')
    if pe:
        if pe < 15:
            valuation_score += 10
            signals.append({
                "category": "Valuation", "metric": "P/E Ratio",
                "value": f"{pe:.1f}x", "signal": "UNDERVALUED", "score": "+10",
                "detail": "Low P/E suggests undervaluation vs market average (~20x)",
                "benchmark": "Market avg: ~20x, Threshold: <15x"
            })
        elif pe < 20:
            valuation_score += 5
            signals.append({
                "category": "Valuation", "metric": "P/E Ratio",
                "value": f"{pe:.1f}x", "signal": "FAIR", "score": "+5",
                "detail": "P/E in line with market average",
                "benchmark": "Market avg: ~20x"
            })
        elif pe < 30:
            valuation_score += 0
            signals.append({
                "category": "Valuation", "metric": "P/E Ratio",
                "value": f"{pe:.1f}x", "signal": "ELEVATED", "score": "0",
                "detail": "Premium valuation, needs growth to justify",
                "benchmark": "Above market avg"
            })
        else:
            valuation_score -= 10
            signals.append({
                "category": "Valuation", "metric": "P/E Ratio",
                "value": f"{pe:.1f}x", "signal": "OVERVALUED", "score": "-10",
                "detail": "High P/E - significant premium to market",
                "benchmark": "Threshold: >30x"
            })

    # PEG Ratio (10 points) — calculate manually if yfinance doesn't provide it
    peg = data.get('peg_ratio')
    if not peg:
        _pe = data.get('pe_ratio')
        _eg = data.get('earnings_growth')
        if _pe and _eg and _eg > 0:
            peg = _pe / (_eg * 100)
    if peg:
        if peg < 1:
            valuation_score += 10
            signals.append({
                "category": "Valuation", "metric": "PEG Ratio",
                "value": f"{peg:.2f}", "signal": "UNDERVALUED", "score": "+10",
                "detail": "PEG < 1 suggests growth not fully priced in",
                "benchmark": "Fair value: PEG = 1"
            })
        elif peg < 1.5:
            valuation_score += 5
            signals.append({
                "category": "Valuation", "metric": "PEG Ratio",
                "value": f"{peg:.2f}", "signal": "FAIR", "score": "+5",
                "detail": "Reasonable valuation for growth",
                "benchmark": "Fair value: PEG = 1-1.5"
            })
        elif peg < 2:
            valuation_score += 0
            signals.append({
                "category": "Valuation", "metric": "PEG Ratio",
                "value": f"{peg:.2f}", "signal": "ELEVATED", "score": "0",
                "detail": "Paying premium for growth",
                "benchmark": "PEG > 1.5"
            })
        else:
            valuation_score -= 5
            signals.append({
                "category": "Valuation", "metric": "PEG Ratio",
                "value": f"{peg:.2f}", "signal": "OVERVALUED", "score": "-5",
                "detail": "Growth fully priced in and more",
                "benchmark": "PEG > 2"
            })

    # EV/EBITDA (10 points)
    ev_ebitda = data.get('ev_ebitda')
    if ev_ebitda:
        if ev_ebitda < 8:
            valuation_score += 10
            signals.append({
                "category": "Valuation", "metric": "EV/EBITDA",
                "value": f"{ev_ebitda:.1f}x", "signal": "UNDERVALUED", "score": "+10",
                "detail": "Low EV/EBITDA - attractive on cash flow basis",
                "benchmark": "Threshold: <8x (value)"
            })
        elif ev_ebitda < 12:
            valuation_score += 5
            signals.append({
                "category": "Valuation", "metric": "EV/EBITDA",
                "value": f"{ev_ebitda:.1f}x", "signal": "FAIR", "score": "+5",
                "detail": "Reasonable EV/EBITDA multiple",
                "benchmark": "Market avg: 10-12x"
            })
        elif ev_ebitda < 18:
            valuation_score += 0
            signals.append({
                "category": "Valuation", "metric": "EV/EBITDA",
                "value": f"{ev_ebitda:.1f}x", "signal": "ELEVATED", "score": "0",
                "detail": "Premium EV/EBITDA multiple",
                "benchmark": "Growth premium: 12-18x"
            })
        else:
            valuation_score -= 5
            signals.append({
                "category": "Valuation", "metric": "EV/EBITDA",
                "value": f"{ev_ebitda:.1f}x", "signal": "EXPENSIVE", "score": "-5",
                "detail": "High EV/EBITDA - rich valuation",
                "benchmark": "Threshold: >18x"
            })

    total_score += valuation_score
    max_score += valuation_max

    # ===== PROFITABILITY (25 points) =====
    profitability_score = 0
    profitability_max = 25

    # Operating Margin (10 points)
    op_margin = data.get('operating_margin')
    if op_margin:
        op_margin_pct = op_margin * 100
        if op_margin_pct > 20:
            profitability_score += 10
            signals.append({
                "category": "Profitability", "metric": "Operating Margin",
                "value": f"{op_margin_pct:.1f}%", "signal": "STRONG", "score": "+10",
                "detail": "Excellent operating efficiency",
                "benchmark": "Elite: >20%"
            })
        elif op_margin_pct > 10:
            profitability_score += 5
            signals.append({
                "category": "Profitability", "metric": "Operating Margin",
                "value": f"{op_margin_pct:.1f}%", "signal": "GOOD", "score": "+5",
                "detail": "Healthy operating margin",
                "benchmark": "Good: 10-20%"
            })
        elif op_margin_pct > 0:
            profitability_score += 0
            signals.append({
                "category": "Profitability", "metric": "Operating Margin",
                "value": f"{op_margin_pct:.1f}%", "signal": "WEAK", "score": "0",
                "detail": "Low operating margin",
                "benchmark": "Below avg: <10%"
            })
        else:
            profitability_score -= 5
            signals.append({
                "category": "Profitability", "metric": "Operating Margin",
                "value": f"{op_margin_pct:.1f}%", "signal": "LOSS", "score": "-5",
                "detail": "Operating at a loss",
                "benchmark": "Negative margin"
            })

    # ROE (10 points)
    roe = data.get('roe')
    if roe:
        roe_pct = roe * 100
        if roe_pct > 20:
            profitability_score += 10
            _de_check = data.get('debt_to_equity')
            _roe_note = (
                "High ROE partly driven by share buybacks reducing book equity — "
                "operating profitability is strong but the metric is amplified by capital structure"
                if (_de_check and _de_check > 10) else "Superior capital efficiency"
            )
            signals.append({
                "category": "Profitability", "metric": "Return on Equity",
                "value": f"{roe_pct:.1f}%", "signal": "EXCELLENT", "score": "+10",
                "detail": _roe_note,
                "benchmark": "Elite: >20%"
            })
        elif roe_pct > 15:
            profitability_score += 7
            signals.append({
                "category": "Profitability", "metric": "Return on Equity",
                "value": f"{roe_pct:.1f}%", "signal": "STRONG", "score": "+7",
                "detail": "Above-average ROE",
                "benchmark": "Good: 15-20%"
            })
        elif roe_pct > 10:
            profitability_score += 3
            signals.append({
                "category": "Profitability", "metric": "Return on Equity",
                "value": f"{roe_pct:.1f}%", "signal": "ADEQUATE", "score": "+3",
                "detail": "Acceptable ROE",
                "benchmark": "Fair: 10-15%"
            })
        else:
            profitability_score += 0
            signals.append({
                "category": "Profitability", "metric": "Return on Equity",
                "value": f"{roe_pct:.1f}%", "signal": "WEAK", "score": "0",
                "detail": "Below-average capital efficiency",
                "benchmark": "Weak: <10%"
            })

    # Free Cash Flow (5 points)
    fcf = data.get('free_cash_flow')
    market_cap = data.get('market_cap')
    if fcf and market_cap and market_cap > 0:
        fcf_yield = fcf / market_cap * 100
        if fcf_yield > 5:
            profitability_score += 5
            signals.append({
                "category": "Profitability", "metric": "FCF Yield",
                "value": f"{fcf_yield:.1f}%", "signal": "STRONG", "score": "+5",
                "detail": "Excellent free cash flow generation",
                "benchmark": "Strong: >5%"
            })
        elif fcf_yield > 2:
            profitability_score += 3
            signals.append({
                "category": "Profitability", "metric": "FCF Yield",
                "value": f"{fcf_yield:.1f}%", "signal": "GOOD", "score": "+3",
                "detail": "Healthy cash flow",
                "benchmark": "Good: 2-5%"
            })
        elif fcf_yield > 0:
            profitability_score += 0
            signals.append({
                "category": "Profitability", "metric": "FCF Yield",
                "value": f"{fcf_yield:.1f}%", "signal": "LOW", "score": "0",
                "detail": "Minimal free cash flow",
                "benchmark": "Low: 0-2%"
            })
        else:
            profitability_score -= 3
            signals.append({
                "category": "Profitability", "metric": "FCF Yield",
                "value": f"{fcf_yield:.1f}%", "signal": "NEGATIVE", "score": "-3",
                "detail": "Cash burn - needs financing",
                "benchmark": "Negative FCF"
            })

    total_score += profitability_score
    max_score += profitability_max

    # ===== GROWTH (25 points) =====
    growth_score = 0
    growth_max = 25

    # Revenue Growth (12 points)
    rev_growth = data.get('revenue_growth')
    if rev_growth:
        rev_growth_pct = rev_growth * 100
        if rev_growth_pct > 25:
            growth_score += 12
            signals.append({
                "category": "Growth", "metric": "Revenue Growth",
                "value": f"{rev_growth_pct:.1f}%", "signal": "HIGH GROWTH", "score": "+12",
                "detail": "Exceptional revenue growth",
                "benchmark": "High growth: >25%"
            })
        elif rev_growth_pct > 10:
            growth_score += 8
            signals.append({
                "category": "Growth", "metric": "Revenue Growth",
                "value": f"{rev_growth_pct:.1f}%", "signal": "SOLID", "score": "+8",
                "detail": "Strong revenue growth",
                "benchmark": "Solid: 10-25%"
            })
        elif rev_growth_pct > 0:
            growth_score += 4
            signals.append({
                "category": "Growth", "metric": "Revenue Growth",
                "value": f"{rev_growth_pct:.1f}%", "signal": "MODERATE", "score": "+4",
                "detail": "Modest revenue growth",
                "benchmark": "Moderate: 0-10%"
            })
        else:
            growth_score -= 5
            signals.append({
                "category": "Growth", "metric": "Revenue Growth",
                "value": f"{rev_growth_pct:.1f}%", "signal": "DECLINING", "score": "-5",
                "detail": "Revenue contraction",
                "benchmark": "Declining: <0%"
            })

    # Earnings Growth (13 points)
    earn_growth = data.get('earnings_growth')
    if earn_growth:
        earn_growth_pct = earn_growth * 100
        if earn_growth_pct > 30:
            growth_score += 13
            signals.append({
                "category": "Growth", "metric": "Earnings Growth",
                "value": f"{earn_growth_pct:.1f}%", "signal": "HIGH GROWTH", "score": "+13",
                "detail": "Exceptional earnings growth",
                "benchmark": "High growth: >30%"
            })
        elif earn_growth_pct > 15:
            growth_score += 8
            signals.append({
                "category": "Growth", "metric": "Earnings Growth",
                "value": f"{earn_growth_pct:.1f}%", "signal": "SOLID", "score": "+8",
                "detail": "Strong earnings growth",
                "benchmark": "Solid: 15-30%"
            })
        elif earn_growth_pct > 0:
            growth_score += 4
            signals.append({
                "category": "Growth", "metric": "Earnings Growth",
                "value": f"{earn_growth_pct:.1f}%", "signal": "MODERATE", "score": "+4",
                "detail": "Modest earnings growth",
                "benchmark": "Moderate: 0-15%"
            })
        else:
            growth_score -= 5
            signals.append({
                "category": "Growth", "metric": "Earnings Growth",
                "value": f"{earn_growth_pct:.1f}%", "signal": "DECLINING", "score": "-5",
                "detail": "Earnings contraction",
                "benchmark": "Declining: <0%"
            })

    total_score += growth_score
    max_score += growth_max

    # ===== FINANCIAL HEALTH (20 points) =====
    health_score = 0
    health_max = 20

    # Debt/Equity (10 points)
    # Note: companies with large buybacks (e.g. Apple) can have negative book equity,
    # making D/E artificially high. Detect this and use Debt/EBITDA instead.
    de_ratio = data.get('debt_to_equity')
    if de_ratio is not None:
        _ev_ebitda = data.get('ev_ebitda')
        _ev = data.get('enterprise_value')
        _total_debt = data.get('total_debt')
        _debt_ebitda = None
        if _ev_ebitda and _ev and _ev_ebitda > 0 and _total_debt:
            _ebitda_est = _ev / _ev_ebitda
            if _ebitda_est > 0:
                _debt_ebitda = _total_debt / _ebitda_est

        _negative_equity = de_ratio > 10

        if _negative_equity and _debt_ebitda is not None:
            if _debt_ebitda < 2:
                health_score += 7
                signals.append({
                    "category": "Financial Health", "metric": "Debt/Equity",
                    "value": f"{de_ratio:.1f}x (see note)", "signal": "MANAGEABLE", "score": "+7",
                    "detail": f"D/E of {de_ratio:.1f}x is distorted by negative book equity from share buybacks. Debt/EBITDA of {_debt_ebitda:.1f}x shows manageable leverage.",
                    "benchmark": "Debt/EBITDA < 2x = healthy leverage"
                })
            elif _debt_ebitda < 4:
                health_score += 3
                signals.append({
                    "category": "Financial Health", "metric": "Debt/Equity",
                    "value": f"{de_ratio:.1f}x (see note)", "signal": "MODERATE", "score": "+3",
                    "detail": f"D/E of {de_ratio:.1f}x distorted by negative book equity (buybacks). Debt/EBITDA of {_debt_ebitda:.1f}x shows moderate leverage.",
                    "benchmark": "Debt/EBITDA 2-4x = moderate leverage"
                })
            else:
                health_score += 0
                signals.append({
                    "category": "Financial Health", "metric": "Debt/Equity",
                    "value": f"{de_ratio:.1f}x (see note)", "signal": "ELEVATED", "score": "0",
                    "detail": f"D/E of {de_ratio:.1f}x distorted by negative book equity (buybacks). Debt/EBITDA of {_debt_ebitda:.1f}x is elevated.",
                    "benchmark": "Debt/EBITDA > 4x = elevated leverage"
                })
        elif de_ratio < 0.3:
            health_score += 10
            signals.append({
                "category": "Financial Health", "metric": "Debt/Equity",
                "value": f"{de_ratio:.2f}x", "signal": "STRONG", "score": "+10",
                "detail": "Very low leverage - conservative balance sheet",
                "benchmark": "Conservative: <0.3"
            })
        elif de_ratio < 0.7:
            health_score += 7
            signals.append({
                "category": "Financial Health", "metric": "Debt/Equity",
                "value": f"{de_ratio:.2f}x", "signal": "HEALTHY", "score": "+7",
                "detail": "Moderate leverage",
                "benchmark": "Healthy: 0.3-0.7"
            })
        elif de_ratio < 1.5:
            health_score += 3
            signals.append({
                "category": "Financial Health", "metric": "Debt/Equity",
                "value": f"{de_ratio:.2f}x", "signal": "ELEVATED", "score": "+3",
                "detail": "Higher leverage - monitor interest coverage",
                "benchmark": "Elevated: 0.7-1.5"
            })
        else:
            health_score -= 5
            signals.append({
                "category": "Financial Health", "metric": "Debt/Equity",
                "value": f"{de_ratio:.2f}x", "signal": "HIGH RISK", "score": "-5",
                "detail": "High leverage - balance sheet risk",
                "benchmark": "High risk: >1.5"
            })

    # Current Ratio (10 points)
    current_ratio = data.get('current_ratio')
    if current_ratio:
        if current_ratio > 2:
            health_score += 10
            signals.append({
                "category": "Financial Health", "metric": "Current Ratio",
                "value": f"{current_ratio:.2f}", "signal": "STRONG", "score": "+10",
                "detail": "Excellent liquidity position",
                "benchmark": "Strong: >2.0"
            })
        elif current_ratio > 1.5:
            health_score += 7
            signals.append({
                "category": "Financial Health", "metric": "Current Ratio",
                "value": f"{current_ratio:.2f}", "signal": "HEALTHY", "score": "+7",
                "detail": "Good liquidity",
                "benchmark": "Healthy: 1.5-2.0"
            })
        elif current_ratio > 1:
            health_score += 3
            signals.append({
                "category": "Financial Health", "metric": "Current Ratio",
                "value": f"{current_ratio:.2f}", "signal": "ADEQUATE", "score": "+3",
                "detail": "Adequate liquidity",
                "benchmark": "Adequate: 1.0-1.5"
            })
        else:
            health_score -= 5
            signals.append({
                "category": "Financial Health", "metric": "Current Ratio",
                "value": f"{current_ratio:.2f}", "signal": "WEAK", "score": "-5",
                "detail": "Liquidity concerns",
                "benchmark": "Weak: <1.0"
            })

    total_score += health_score
    max_score += health_max

    # Calculate percentages
    score_pct = (total_score / max_score * 100) if max_score > 0 else 0

    # Remap to 0-100 display scale where 50 = neutral
    # score_pct ranges from ~-40 to +100; map via 50 + x*0.5 then clamp
    display_score = max(0.0, min(100.0, 50.0 + score_pct * 0.5))

    # Determine fundamental rating based on display_score (0-100 scale)
    if display_score >= 80:
        fund_rating = ("STRONG BUY", "#00ff00")
    elif display_score >= 60:
        fund_rating = ("BUY", "#3fb950")
    elif display_score >= 40:
        fund_rating = ("HOLD", "#d29922")
    elif display_score >= 20:
        fund_rating = ("UNDERPERFORM", "#f85149")
    else:
        fund_rating = ("SELL", "#ff0000")

    return {
        "signals": signals,
        "total_score": total_score,
        "max_score": max_score,
        "score_pct": score_pct,
        "display_score": display_score,
        "rating": fund_rating[0],
        "rating_color": fund_rating[1],
        "breakdown": {
            "valuation": {"score": valuation_score, "max": valuation_max},
            "profitability": {"score": profitability_score, "max": profitability_max},
            "growth": {"score": growth_score, "max": growth_max},
            "health": {"score": health_score, "max": health_max}
        }
    }


# ============== VALUATION MODEL ==============
def calculate_valuation(data: Dict, fundamental_analysis: Dict) -> Dict:
    """Calculate price targets using multiple valuation methods."""
    price = data['price']
    results = {}

    # ===== RELATIVE VALUATION =====
    pe = data.get('pe_ratio')
    eps = data.get('eps')
    forward_eps = data.get('forward_eps')

    # P/E Based Valuation
    if pe and eps:
        # Assume sector average P/E of 18-22
        sector_pe_low = 16
        sector_pe_mid = 20
        sector_pe_high = 25

        pe_target_low = eps * sector_pe_low
        pe_target_mid = eps * sector_pe_mid
        pe_target_high = eps * sector_pe_high

        results['pe_valuation'] = {
            'low': pe_target_low,
            'mid': pe_target_mid,
            'high': pe_target_high,
            'current_pe': pe
        }

    # Forward P/E Valuation
    if forward_eps:
        fwd_pe_low = 14
        fwd_pe_mid = 18
        fwd_pe_high = 22

        fwd_target_low = forward_eps * fwd_pe_low
        fwd_target_mid = forward_eps * fwd_pe_mid
        fwd_target_high = forward_eps * fwd_pe_high

        results['forward_pe_valuation'] = {
            'low': fwd_target_low,
            'mid': fwd_target_mid,
            'high': fwd_target_high,
            'forward_eps': forward_eps
        }

    # ===== ANALYST CONSENSUS =====
    if data.get('target_price'):
        results['analyst_target'] = {
            'low': data.get('target_low', data['target_price'] * 0.85),
            'mid': data['target_price'],
            'high': data.get('target_high', data['target_price'] * 1.15),
            'num_analysts': data.get('num_analysts', 0)
        }

    # ===== SIMPLIFIED DCF =====
    fcf = data.get('free_cash_flow')
    shares = data.get('shares_outstanding')

    if fcf and shares and fcf > 0:
        fcf_per_share = fcf / shares

        # Scenarios: Bear (5% growth, 12% discount), Base (8% growth, 10% discount), Bull (12% growth, 8% discount)
        def simple_dcf(fcf_ps, growth_rate, discount_rate, terminal_growth=0.025, years=5):
            total_pv = 0
            for year in range(1, years + 1):
                future_fcf = fcf_ps * (1 + growth_rate) ** year
                pv = future_fcf / (1 + discount_rate) ** year
                total_pv += pv
            # Terminal value
            terminal_fcf = fcf_ps * (1 + growth_rate) ** years * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            terminal_pv = terminal_value / (1 + discount_rate) ** years
            return total_pv + terminal_pv

        _rev_g = (data.get('revenue_growth') or 0)
        _dcf_g_base = float(max(min(_rev_g, 0.18), 0.03))
        _dcf_g_bull = float(min(_dcf_g_base * 1.35, 0.25))
        _dcf_g_bear = float(max(_dcf_g_base * 0.50, 0.02))

        dcf_bear = simple_dcf(fcf_per_share, _dcf_g_bear, 0.12)
        dcf_base = simple_dcf(fcf_per_share, _dcf_g_base, 0.10)
        dcf_bull = simple_dcf(fcf_per_share, _dcf_g_bull, 0.08)

        results['dcf_valuation'] = {
            'bear': dcf_bear,
            'base': dcf_base,
            'bull': dcf_bull,
            'fcf_per_share': fcf_per_share,
            'growth_bear': _dcf_g_bear,
            'growth_base': _dcf_g_base,
            'growth_bull': _dcf_g_bull,
        }

    # ===== COMPOSITE TARGET =====
    all_mids = []
    all_lows = []
    all_highs = []

    if 'pe_valuation' in results:
        all_mids.append(results['pe_valuation']['mid'])
        all_lows.append(results['pe_valuation']['low'])
        all_highs.append(results['pe_valuation']['high'])

    if 'forward_pe_valuation' in results:
        all_mids.append(results['forward_pe_valuation']['mid'])
        all_lows.append(results['forward_pe_valuation']['low'])
        all_highs.append(results['forward_pe_valuation']['high'])

    if 'analyst_target' in results:
        all_mids.append(results['analyst_target']['mid'])
        all_lows.append(results['analyst_target']['low'])
        all_highs.append(results['analyst_target']['high'])

    if 'dcf_valuation' in results:
        all_mids.append(results['dcf_valuation']['base'])
        all_lows.append(results['dcf_valuation']['bear'])
        all_highs.append(results['dcf_valuation']['bull'])

    if all_mids:
        results['composite'] = {
            'target_low': np.mean(all_lows),
            'target_mid': np.mean(all_mids),
            'target_high': np.mean(all_highs),
            'upside_low': (np.mean(all_lows) - price) / price * 100,
            'upside_mid': (np.mean(all_mids) - price) / price * 100,
            'upside_high': (np.mean(all_highs) - price) / price * 100
        }

    return results


# ============== RETURN FORECASTING ==============
def forecast_returns(data: Dict, tech_analysis: Dict, fund_analysis: Dict, valuation: Dict) -> Dict:
    """Generate expected returns with confidence intervals."""
    price = data['price']
    beta = data.get('beta', 1) or 1

    # Base expected return from valuation
    if 'composite' in valuation:
        base_annual_return = valuation['composite']['upside_mid']
    else:
        base_annual_return = 0

    # Adjust based on technical score (score_pct -100 to +100 → ±5%)
    tech_adjustment = tech_analysis['score_pct'] / 100 * 5

    # Adjust based on fundamental score (display_score 0-100 → -5% to +5%)
    _fund_disp = fund_analysis.get('display_score', max(0.0, min(100.0, 50.0 + fund_analysis['score_pct'] * 0.5)))
    fund_adjustment = (_fund_disp - 50) / 50 * 5

    # Combined expected annual return
    expected_annual = base_annual_return + tech_adjustment + fund_adjustment

    # Calculate historical volatility for confidence intervals
    hist = data.get('hist_1y')
    if hist is not None and len(hist) > 20:
        daily_returns = hist['Close'].pct_change().dropna()
        daily_vol = daily_returns.std()
        annual_vol = daily_vol * np.sqrt(252) * 100
    else:
        annual_vol = 25  # Default assumption

    # Pre-compute confidence metrics (signal consistency + valuation alignment)
    _tech_disp_fc = (tech_analysis['score_pct'] + 100) / 2
    _fund_disp_fc = fund_analysis.get('display_score', max(0.0, min(100.0, 50.0 + fund_analysis['score_pct'] * 0.5)))
    _combined_disp_fc = (_tech_disp_fc + _fund_disp_fc) / 2
    _dist_from_neutral = abs(_combined_disp_fc - 50)

    # Signal consistency: fraction of signals pointing in one direction
    _n_bull_fc = sum(1 for s in tech_analysis['signals'] + fund_analysis['signals']
                     if s.get('signal') in ['BULLISH', 'STRONG', 'UNDERVALUED', 'HIGH GROWTH', 'EXCELLENT'])
    _n_bear_fc = sum(1 for s in tech_analysis['signals'] + fund_analysis['signals']
                     if s.get('signal') in ['BEARISH', 'WEAK', 'OVERVALUED', 'DECLINING', 'HIGH RISK', 'LOSS'])
    _total_sig_fc = _n_bull_fc + _n_bear_fc
    _signal_consistency = abs(_n_bull_fc - _n_bear_fc) / _total_sig_fc if _total_sig_fc > 0 else 0

    # Valuation alignment: does target direction match score direction?
    _target_upside_fc = valuation['composite']['upside_mid'] if 'composite' in valuation else 0
    _score_bullish = _combined_disp_fc > 50
    _target_bullish = _target_upside_fc > 0
    _valuation_aligned = (_score_bullish == _target_bullish)

    # Composite confidence score
    _conf_score = _dist_from_neutral
    if _signal_consistency > 0.5:
        _conf_score += 10   # signals predominantly point one way
    if _valuation_aligned:
        _conf_score += 5    # score direction matches target direction
    if abs(_target_upside_fc) > 15:
        _conf_score += 5    # strong valuation signal (large gap between price and target)

    if _conf_score > 25:
        _base_confidence = "High"
    elif _conf_score > 12:
        _base_confidence = "Medium"
    else:
        _base_confidence = "Low"

    # Time-scaled forecasts
    forecasts = {}

    for period, days, name in [(7, 7, '1 Week'), (30, 30, '1 Month'), (90, 90, '3 Months'), (180, 180, '6 Months')]:
        # Scale expected return and volatility
        period_return = expected_annual * (days / 365)
        period_vol = annual_vol * np.sqrt(days / 252)

        # Point estimate
        point_estimate = period_return

        # Range (1 std dev)
        range_low = point_estimate - period_vol
        range_high = point_estimate + period_vol

        # Fix 6: Confidence based on signal consistency, valuation alignment, and directional score
        confidence = _base_confidence
        if _base_confidence == "High":
            probability = "65-75%" if expected_annual >= 0 else "25-35%"
        elif _base_confidence == "Medium":
            probability = "55-65%" if expected_annual >= 0 else "35-45%"
        else:
            probability = "45-55%"

        # Price targets
        price_target = price * (1 + point_estimate / 100)
        price_low = price * (1 + range_low / 100)
        price_high = price * (1 + range_high / 100)

        forecasts[name] = {
            'point_estimate': point_estimate,
            'range_low': range_low,
            'range_high': range_high,
            'price_target': price_target,
            'price_low': price_low,
            'price_high': price_high,
            'confidence': confidence,
            'probability': probability,
            'volatility': period_vol
        }

    return forecasts


# ============== FINAL RECOMMENDATION ==============
def generate_recommendation(data: Dict, tech_analysis: Dict, fund_analysis: Dict, valuation: Dict, forecasts: Dict,
                            supports=None, resistances=None) -> Dict:
    """Generate final institutional-grade recommendation."""
    price = data['price']

    # Combined score on 0-100 display scale (50 = neutral)
    # Use display_score for fundamentals and remap tech score_pct similarly
    fund_display = fund_analysis.get('display_score', max(0.0, min(100.0, 50.0 + fund_analysis['score_pct'] * 0.5)))
    tech_display = max(0.0, min(100.0, (tech_analysis['score_pct'] + 100) / 2))
    combined_score = fund_display * 0.6 + tech_display * 0.4

    # Determine action based on 0-100 display scale (80/60/40/20/0)
    if combined_score >= 80:
        action = "STRONG BUY"
        action_color = "#00ff00"
    elif combined_score >= 60:
        action = "BUY"
        action_color = "#3fb950"
    elif combined_score >= 40:
        action = "HOLD"
        action_color = "#d29922"
    elif combined_score >= 20:
        action = "UNDERPERFORM"
        action_color = "#f85149"
    else:
        action = "SELL"
        action_color = "#ff0000"

    # Trade decision
    if action in ["STRONG BUY", "BUY"]:
        trade_decision = "INITIATE LONG POSITION"
    elif action in ["UNDERPERFORM", "SELL"]:
        trade_decision = "REDUCE / AVOID POSITION"
    else:
        trade_decision = "HOLD - WAIT FOR CLEARER SIGNAL"

    # Price target and upside (enforce Bear ≤ Base ≤ Bull)
    if 'composite' in valuation:
        target_price = valuation['composite']['target_mid']
        upside = valuation['composite']['upside_mid']
        target_low = valuation['composite']['target_low']
        target_high = valuation['composite']['target_high']
        # Enforce ordering: bear ≤ base ≤ bull
        sorted_targets = sorted([target_low, target_price, target_high])
        target_low, target_price, target_high = sorted_targets
        upside = (target_price - price) / price * 100
        upside_high = (target_high - price) / price * 100
    else:
        target_price = price
        upside = 0
        target_low = price * 0.9
        target_high = price * 1.1
        upside_high = 10.0

    # Sanity check / Fix 1+2: Override to SELL when base target >10% below price AND bull case <5% upside.
    # Also cap composite score to ≤35 (<40 = SELL range) so rating and score are always consistent.
    if upside < -10 and upside_high < 5:
        action = "SELL"
        action_color = "#ff0000"
        trade_decision = "REDUCE / AVOID POSITION"
        combined_score = min(combined_score, 35)  # keep score in SELL/UNDERPERFORM range

    # Key drivers (bullish factors)
    bullish_drivers = []
    bearish_risks = []
    roe_capital_note = None  # ROE note when buyback-driven (Fix 4)

    roe = data.get('roe', 0) or 0
    cr = data.get('current_ratio', 0) or 0

    for signal in tech_analysis['signals'] + fund_analysis['signals']:
        metric_name = signal.get('metric', signal.get('indicator', 'N/A'))
        sig_type = signal.get('signal', '')

        if sig_type in ['BULLISH', 'STRONG', 'UNDERVALUED', 'HIGH GROWTH', 'EXCELLENT']:
            # Fix 4: ROE >100% is buyback-driven — move to capital structure note, not bullish driver
            if 'ROE' in metric_name and roe * 100 > 100:
                roe_capital_note = (
                    f"ROE of {roe*100:.0f}% is largely buyback-driven (aggressive share repurchases "
                    f"have reduced book equity); reflects capital return policy rather than pure operational efficiency."
                )
                continue
            bullish_drivers.append(f"{metric_name}: {signal.get('detail', '')}")

        elif sig_type in ['BEARISH', 'WEAK', 'OVERVALUED', 'DECLINING', 'HIGH RISK', 'LOSS']:
            # Fix 3: Exclude Current Ratio from risks when company has strong FCF (cr > 0.5)
            if 'Current Ratio' in metric_name and cr > 0.5:
                continue
            bearish_risks.append(f"{metric_name}: {signal.get('detail', '')}")

    # Fix 2: Valuation description — corrected ternary and escaped $ to prevent LaTeX rendering
    if upside > 15:
        valuation_desc = "represents an attractive entry point"
    elif upside > -10:
        valuation_desc = "is fairly valued"
    else:
        valuation_desc = "appears overvalued relative to intrinsic value"

    capital_note_section = ""
    if roe_capital_note:
        capital_note_section = f"\n**Capital Structure Note:**\n{roe_capital_note}\n"

    # Fix 2: Use \$ to prevent Streamlit markdown from rendering dollar signs as LaTeX
    rationale = f"""
**Investment Thesis:**
{data['name']} ({data['ticker']}) rates as a **{action}** based on combined technical and fundamental analysis.

**Key Drivers:**
{''.join(['• ' + d + chr(10) for d in bullish_drivers[:3]]) if bullish_drivers else '• No significant bullish factors identified'}

**Key Risks:**
{''.join(['• ' + r + chr(10) for r in bearish_risks[:3]]) if bearish_risks else '• No significant bearish factors identified'}
{capital_note_section}
**Valuation:**
Current price of \${price:.2f} {valuation_desc} with a base case target of \${target_price:.2f} ({upside:+.1f}% potential). Bull case target: \${target_high:.2f} ({upside_high:+.1f}%).
"""

    # Fix 5: Use actual support/resistance levels from technical analysis
    support_below = None
    resistance_above = None
    if supports:
        below = [s for s in supports if s < price]
        if below:
            support_below = max(below)
    if resistances:
        above = [r for r in resistances if r > price]
        if above:
            resistance_above = min(above)

    if action in ["STRONG BUY", "BUY"]:
        if support_below:
            pct = (support_below / price - 1) * 100
            invalidation = f"Exit if price breaks below key support at \${support_below:.2f} ({pct:+.1f}%) on a closing basis."
        else:
            invalidation = f"Exit if price falls below \${price * 0.9:.2f} (-10%) or if key technical support breaks."
    elif action in ["UNDERPERFORM", "SELL"]:
        if resistance_above:
            pct = (resistance_above / price - 1) * 100
            invalidation = f"Reconsider if price breaks above key resistance at \${resistance_above:.2f} ({pct:+.1f}%) with strong volume."
        else:
            invalidation = f"Reconsider if price breaks above \${price * 1.1:.2f} (+10%) with strong volume."
    else:
        if support_below and resistance_above:
            invalidation = (f"Monitor range \${support_below:.2f} – \${resistance_above:.2f}. "
                            f"Decisive close outside this band signals next directional move.")
        else:
            invalidation = "Monitor for decisive breakout above resistance or breakdown below support."

    return {
        'action': action,
        'action_color': action_color,
        'trade_decision': trade_decision,
        'combined_score': combined_score,
        'target_price': target_price,
        'target_low': target_low,
        'target_high': target_high,
        'upside': upside,
        'rationale': rationale,
        'bullish_drivers': bullish_drivers[:5],
        'bearish_risks': bearish_risks[:5],
        'invalidation': invalidation
    }


# ============== CHAT CONTEXT BUILDER ==============
def build_analysis_context(data: Dict, tech_analysis: Dict, fund_analysis: Dict,
                           valuation: Dict, forecasts: Dict, recommendation: Dict) -> str:
    """Build comprehensive context string for the AI chat."""
    ctx = f"""STOCK ANALYSIS REPORT - {data['name']} ({data['ticker']})
Data Source: Yahoo Finance (real-time market data, SEC filings 10-K/10-Q, analyst estimates)

=== PRICE DATA ===
Current Price: ${data['price']:.2f}
Previous Close: ${data.get('prev_close', 0):.2f}
52-Week High: ${data.get('high_52w', 0):.2f}
52-Week Low: ${data.get('low_52w', 0):.2f}
Market Cap: ${data.get('market_cap', 0)/1e9:.1f}B
Enterprise Value: ${data.get('enterprise_value', 0)/1e9:.1f}B
Beta: {data.get('beta', 'N/A')}
Sector: {data.get('sector', 'N/A')}
Industry: {data.get('industry', 'N/A')}

=== TECHNICAL ANALYSIS (Score: {tech_analysis['total_score']}/{tech_analysis['max_score']} = {tech_analysis['score_pct']:.0f}%) ===
Rating: {tech_analysis['rating']}
Breakdown:
- Trend: {tech_analysis['breakdown']['trend']['score']}/{tech_analysis['breakdown']['trend']['max']} (MA alignment, price position, golden/death cross)
- Momentum: {tech_analysis['breakdown']['momentum']['score']}/{tech_analysis['breakdown']['momentum']['max']} (RSI, MACD)
- Volatility: {tech_analysis['breakdown']['volatility']['score']}/{tech_analysis['breakdown']['volatility']['max']} (Bollinger Bands, ATR)
- Volume: {tech_analysis['breakdown']['volume']['score']}/{tech_analysis['breakdown']['volume']['max']} (Volume surge, OBV trend)

Technical Signals:
"""
    for s in tech_analysis['signals']:
        ctx += f"- {s.get('indicator', 'N/A')}: {s['signal']} ({s['score']}) - {s['detail']} [Threshold: {s.get('threshold', 'N/A')}]\n"

    ctx += f"""
=== FUNDAMENTAL ANALYSIS (Score: {fund_analysis['total_score']}/{fund_analysis['max_score']} = {fund_analysis['score_pct']:.0f}%) ===
Rating: {fund_analysis['rating']}
Breakdown:
- Valuation: {fund_analysis['breakdown']['valuation']['score']}/{fund_analysis['breakdown']['valuation']['max']} (P/E, PEG, EV/EBITDA)
- Profitability: {fund_analysis['breakdown']['profitability']['score']}/{fund_analysis['breakdown']['profitability']['max']} (Operating margin, ROE, FCF yield)
- Growth: {fund_analysis['breakdown']['growth']['score']}/{fund_analysis['breakdown']['growth']['max']} (Revenue growth, earnings growth)
- Health: {fund_analysis['breakdown']['health']['score']}/{fund_analysis['breakdown']['health']['max']} (Debt/equity, current ratio)

Key Metrics (from Yahoo Finance / SEC filings):
- P/E Ratio: {data.get('pe_ratio', 'N/A')}
- Forward P/E: {data.get('forward_pe', 'N/A')}
- PEG Ratio: {data.get('peg_ratio', 'N/A')}
- P/B Ratio: {data.get('pb_ratio', 'N/A')}
- EV/EBITDA: {data.get('ev_ebitda', 'N/A')}
- Gross Margin: {f"{data['gross_margin']*100:.1f}%" if data.get('gross_margin') else 'N/A'}
- Operating Margin: {f"{data['operating_margin']*100:.1f}%" if data.get('operating_margin') else 'N/A'}
- Net Margin: {f"{data['profit_margin']*100:.1f}%" if data.get('profit_margin') else 'N/A'}
- ROE: {f"{data['roe']*100:.1f}%" if data.get('roe') else 'N/A'}
- ROA: {f"{data['roa']*100:.1f}%" if data.get('roa') else 'N/A'}
- Revenue Growth: {f"{data['revenue_growth']*100:.1f}%" if data.get('revenue_growth') else 'N/A'}
- Earnings Growth: {f"{data['earnings_growth']*100:.1f}%" if data.get('earnings_growth') else 'N/A'}
- EPS: {f"${data['eps']:.2f}" if data.get('eps') else 'N/A'}
- Forward EPS: {f"${data['forward_eps']:.2f}" if data.get('forward_eps') else 'N/A'}
- Debt/Equity: {data.get('debt_to_equity', 'N/A')}
- Current Ratio: {data.get('current_ratio', 'N/A')}
- Free Cash Flow: {f"${data['free_cash_flow']/1e9:.1f}B" if data.get('free_cash_flow') else 'N/A'}
- Dividend Yield: {f"{data['dividend_yield']*100:.2f}%" if data.get('dividend_yield') else 'N/A'}

Fundamental Signals:
"""
    for s in fund_analysis['signals']:
        ctx += f"- {s.get('metric', 'N/A')}: {s['value']} - {s['signal']} ({s['score']}) - {s['detail']} [Benchmark: {s.get('benchmark', 'N/A')}]\n"

    ctx += "\n=== VALUATION MODELS ===\n"
    if 'pe_valuation' in valuation:
        v = valuation['pe_valuation']
        ctx += f"P/E Based: Bear ${v['low']:.2f} | Base ${v['mid']:.2f} | Bull ${v['high']:.2f}\n"
    if 'forward_pe_valuation' in valuation:
        v = valuation['forward_pe_valuation']
        ctx += f"Forward P/E: Bear ${v['low']:.2f} | Base ${v['mid']:.2f} | Bull ${v['high']:.2f}\n"
    if 'analyst_target' in valuation:
        v = valuation['analyst_target']
        ctx += f"Analyst Consensus ({v['num_analysts']} analysts): Low ${v['low']:.2f} | Mean ${v['mid']:.2f} | High ${v['high']:.2f}\n"
    if 'dcf_valuation' in valuation:
        v = valuation['dcf_valuation']
        ctx += f"DCF Model (FCF/share ${v['fcf_per_share']:.2f}): Bear ${v['bear']:.2f} | Base ${v['base']:.2f} | Bull ${v['bull']:.2f}\n"
    if 'composite' in valuation:
        v = valuation['composite']
        ctx += f"COMPOSITE TARGET: ${v['target_mid']:.2f} (Range: ${v['target_low']:.2f} - ${v['target_high']:.2f}) | Upside: {v['upside_mid']:+.1f}%\n"

    ctx += f"""
=== RETURN FORECASTS ===
"""
    for period, f in forecasts.items():
        ctx += f"- {period}: {f['point_estimate']:+.1f}% expected (Range: {f['range_low']:+.1f}% to {f['range_high']:+.1f}%) | Target: ${f['price_target']:.2f} | Confidence: {f['confidence']} ({f['probability']})\n"

    ctx += f"""
=== FINAL RECOMMENDATION ===
Action: {recommendation['action']}
Combined Score: {recommendation['combined_score']:.0f} (60% fundamental + 40% technical)
Target Price: ${recommendation['target_price']:.2f}
Upside/Downside: {recommendation['upside']:+.1f}%
Trade Decision: {recommendation['trade_decision']}

Bullish Factors:
"""
    for d in recommendation.get('bullish_drivers', []):
        ctx += f"- {d}\n"
    ctx += "\nBearish Risks:\n"
    for r in recommendation.get('bearish_risks', []):
        ctx += f"- {r}\n"
    ctx += f"\nInvalidation: {recommendation['invalidation']}\n"

    ctx += """
=== SCORING METHODOLOGY ===
Technical Score (100 points max): 30% Trend + 30% Momentum + 20% Volatility + 20% Volume
Fundamental Score (100 points max): 30% Valuation + 25% Profitability + 25% Growth + 20% Financial Health
Combined Score: 60% Fundamental + 40% Technical
Thresholds: STRONG BUY (>=35) | BUY (>=15) | HOLD (-5 to 15) | SELL (-25 to -5) | STRONG SELL (<-25)
"""
    return ctx


# ============== MAIN DISPLAY FUNCTION ==============
def show_stock_analyzer():
    """Main function to display the institutional stock analyzer."""

    # Back button and header
    col_back, col_title = st.columns([1, 11])
    with col_back:
        st.markdown('<div class="btn-back"></div>', unsafe_allow_html=True)
        if st.button("← Back", key="back_analyzer"):
            st.session_state.current_view = 'home'
            st.rerun()
    with col_title:
        st.write("# 📈 Stock Analyzer")
        st.caption("Technical & Fundamental Analysis with Price Forecasting")

    if 'inst_data' not in st.session_state:
        st.session_state.inst_data = None

    # ── PRE-COMPUTE (uses session state from previous run) ──────────────────
    data = None
    tech_df = tech_analysis = fund_analysis = valuation = forecasts = recommendation = supports = resistances = None
    has_data = st.session_state.inst_data and st.session_state.inst_data.get("valid")

    if has_data:
        data = st.session_state.inst_data
        hist = data['hist_1y']
        tech_df = calculate_technical_indicators(hist)
        tech_analysis = generate_technical_signals(data, tech_df)
        fund_analysis = analyze_fundamentals(data)
        valuation = calculate_valuation(data, fund_analysis)
        forecasts = forecast_returns(data, tech_analysis, fund_analysis, valuation)
        supports, resistances = identify_support_resistance(hist)
        recommendation = generate_recommendation(data, tech_analysis, fund_analysis, valuation, forecasts,
                                                 supports=supports, resistances=resistances)

    # ── TWO-COLUMN LAYOUT: LEFT (input + company info) | RIGHT (analysis tabs) ─
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:16px 14px 10px 14px;margin-bottom:12px;">
  <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Stock Ticker</div>""",
            unsafe_allow_html=True)
        ticker = st.text_input("Ticker Symbol", value="AAPL", key="inst_ticker",
                               placeholder="e.g. AAPL, MSFT, NVDA", label_visibility="collapsed")
        ticker = ticker.strip().upper()
        st.markdown('<div class="btn-rfa"></div>', unsafe_allow_html=True)
        analyze_btn = st.button("Run Full Analysis", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if analyze_btn and ticker:
            with st.spinner(f"Fetching institutional data for {ticker}\u2026"):
                st.session_state.inst_data = fetch_comprehensive_data(ticker)
                st.rerun()

        # ── AI CHAT FORM ─────────────────────────────────────────────────────────
        _chat_key = f"chat_history_{data['ticker']}" if has_data else "chat_history_default"
        if _chat_key not in st.session_state:
            st.session_state[_chat_key] = []

        st.markdown("""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:14px 14px 10px 14px;margin-bottom:10px;">
  <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:8px;">Ask the AI Analyst</div>""",
            unsafe_allow_html=True)
        with st.form("chat_form", clear_on_submit=True):
            _prompt = st.text_input("q", label_visibility="collapsed",
                                    placeholder="e.g. What\u2019s the risk/reward here?")
            _submitted = st.form_submit_button("Send \u2192", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ── AI CHAT HISTORY ──────────────────────────────────────────────────────
        if has_data:
            _chat_sub = f"<span style='font-size:11px;color:#6e7681;'>Chatting about <strong style='color:#58a6ff;'>{data['ticker']} \u2014 {data['name']}</strong></span>"
        else:
            _chat_sub = "<span style='font-size:11px;color:#6e7681;'>Run an analysis to start chatting.</span>"
        st.markdown(f"""
<div style="font-size:13px;font-weight:600;color:#e6edf3;margin:6px 0 6px 0;">
  \U0001f4ac AI Financial Analyst<br>{_chat_sub}
</div>""", unsafe_allow_html=True)

        _chat_box = st.container(height=320)
        # Track whether we had messages BEFORE this render pass so we can
        # clear the empty-state placeholder immediately on first submission.
        _had_msgs_before = bool(st.session_state[_chat_key])
        _empty_state_ph = None
        with _chat_box:
            if not _had_msgs_before:
                _empty_state_ph = st.empty()
                _empty_state_ph.markdown(
                    "<div style='color:#6e7681;font-size:12px;text-align:center;padding-top:36px;'>"
                    "No messages yet<br>"
                    "<span style='font-size:11px;'>Ask a question above to get started.</span></div>",
                    unsafe_allow_html=True
                )
            for _msg in st.session_state[_chat_key]:
                with st.chat_message(_msg["role"]):
                    st.markdown(_msg["content"])

        if st.session_state.get(_chat_key):
            st.markdown('<div class="btn-clear"></div>', unsafe_allow_html=True)
            if st.button("Clear Chat", key="clear_chat_top", use_container_width=True):
                st.session_state[_chat_key] = []
                st.rerun()


        # ── FORM SUBMISSION PROCESSING ───────────────────────────────────────────
        if _submitted and _prompt:
            st.session_state[_chat_key].append({"role": "user", "content": _prompt})
            # Clear the "No messages yet" placeholder immediately so it doesn't
            # coexist with the first real message in the same render pass.
            if _empty_state_ph is not None:
                _empty_state_ph.empty()
            with _chat_box:
                with st.chat_message("user"):
                    st.markdown(_prompt)

            if not has_data:
                _reply = "Please enter a ticker and run an analysis first."
                st.session_state[_chat_key].append({"role": "assistant", "content": _reply})
                with _chat_box:
                    with st.chat_message("assistant"):
                        st.markdown(_reply)
            else:
                _context = build_analysis_context(
                    data, tech_analysis, fund_analysis, valuation, forecasts, recommendation
                )
                _system_msg = (
                    f"You are an expert CFA-certified financial analyst assistant with access to a "
                    f"comprehensive analysis of {data['name']} ({data['ticker']}).\n\n"
                    "Answer questions using the provided analysis data from Yahoo Finance. "
                    "Rules: cite specific numbers, explain reasoning, give quantitative estimates for "
                    "scenarios, note data limitations, do NOT give personal investment advice.\n\n"
                    f"Full analysis:\n\n{_context}"
                )
                try:
                    from openai import OpenAI as _OpenAI
                    _client = _OpenAI(api_key=openai_api_key)
                    _api_msgs = [{"role": "system", "content": _system_msg}]
                    for _m in st.session_state[_chat_key]:
                        _api_msgs.append({"role": _m["role"], "content": _m["content"]})

                    _response_text = ""
                    with _chat_box:
                        with st.chat_message("assistant"):
                            _stream = _client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=_api_msgs,
                                max_tokens=1500,
                                temperature=0.7,
                                stream=True,
                            )
                            _placeholder = st.empty()
                            for _chunk in _stream:
                                if _chunk.choices[0].delta.content is not None:
                                    _response_text += _chunk.choices[0].delta.content
                                    _placeholder.markdown(_response_text + "\u258c")
                            _placeholder.markdown(_response_text)

                    st.session_state[_chat_key].append({"role": "assistant", "content": _response_text})

                except Exception as _e:
                    _reply = f"Chat error: {_e}"
                    st.session_state[_chat_key].append({"role": "assistant", "content": _reply})
                    with _chat_box:
                        with st.chat_message("assistant"):
                            st.markdown(_reply)


    with col_right:
        tab_profile, tab_tech, tab_fund, tab_conclusion = st.tabs(
            ["🏢 Profile", "📊 Technical Analysis", "📋 Fundamental Analysis", "🎯 Conclusion & Forecast"]
        )

        # ── PROFILE TAB ────────────────────────────────────────────────────────
        with tab_profile:
            if not has_data:
                st.info("Enter a ticker and click **Run Full Analysis** to see the profile.")
            else:
                change_pct = ((data['price'] - data.get('prev_close', data['price'])) / data.get('prev_close', data['price']) * 100) if data.get('prev_close') else 0
                price_arrow = "\u25b2" if change_pct >= 0 else "\u25bc"
                price_c = "#3fb950" if change_pct >= 0 else "#f85149"
                pos = (data['price'] - data['low_52w']) / (data['high_52w'] - data['low_52w']) if data['high_52w'] > data['low_52w'] else 0.5
                pos_pct = min(100, max(0, pos * 100))
                _biz = data.get('info', {}).get('longBusinessSummary', '')
                _biz_short = '. '.join(_biz.replace('\n', ' ').split('. ')[:3]) + '.' if _biz else ''
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:20px 18px;margin-bottom:10px;">
  <div style="font-size:20px;font-weight:700;color:#e6edf3 !important;">{data['name']}</div>
  <div style="font-size:12px;color:#8b949e !important;margin-top:2px;">{data['sector']} &middot; {data['industry']}</div>
  <div style="display:flex;gap:16px;margin-top:16px;flex-wrap:wrap;">
    <div style="flex:1;min-width:100px;">
      <div style="font-size:11px;color:#8b949e !important;text-transform:uppercase;letter-spacing:.5px;">Price</div>
      <div style="font-size:26px;font-weight:700;color:#e6edf3 !important;">${data['price']:.2f}</div>
      <div style="font-size:13px;color:{price_c} !important;font-weight:600;">{price_arrow} {change_pct:+.2f}%</div>
    </div>
    <div style="flex:1;min-width:100px;">
      <div style="font-size:11px;color:#8b949e !important;text-transform:uppercase;letter-spacing:.5px;">Market Cap</div>
      <div style="font-size:20px;font-weight:600;color:#e6edf3 !important;">${data['market_cap']/1e9:.1f}B</div>
    </div>
  </div>
  <div style="margin-top:16px;">
    <div style="display:flex;justify-content:space-between;font-size:12px;color:#8b949e !important;margin-bottom:6px;">
      <span style="color:#8b949e !important;">${data['low_52w']:.0f} low</span>
      <span style="color:#c9d1d9 !important;font-weight:600;">52-Week Range</span>
      <span style="color:#8b949e !important;">${data['high_52w']:.0f} high</span>
    </div>
    <div style="background:#21262d;border-radius:4px;height:8px;position:relative;">
      <div style="background:#58a6ff;width:{pos_pct:.0f}%;height:8px;border-radius:4px;"></div>
    </div>
    <div style="text-align:center;font-size:11px;color:#6e7681 !important;margin-top:4px;">{pos_pct:.0f}% of annual range</div>
  </div>
</div>""", unsafe_allow_html=True)
                if _biz_short:
                    st.markdown(f"""
<div style="background:#0d1f38;border:1px solid #1d4ed8;border-radius:8px;padding:14px 16px;margin-top:4px;">
  <div style="font-size:10px;color:#3b82f6 !important;text-transform:uppercase;letter-spacing:1px;font-weight:700;margin-bottom:8px;">About the Company</div>
  <div style="font-size:13px;color:#8b9db5 !important;line-height:1.65;">{_biz_short}</div>
</div>""", unsafe_allow_html=True)

                # ── KEY STATS CARD ─────────────────────────────────────────────
                st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                _pe_ks   = data.get('pe_ratio')
                _fpe_ks  = data.get('forward_pe')
                _eps_ks  = data.get('eps')
                _rev_ks  = data.get('info', {}).get('totalRevenue', 0) or 0
                _div_ks  = data.get('info', {}).get('dividendYield', None)
                _beta_ks = data.get('info', {}).get('beta', None)
                _div_str = f"{_div_ks*100:.2f}%" if _div_ks else "—"
                _beta_str = f"{_beta_ks:.2f}" if _beta_ks is not None else "N/A"
                _rev_str = f"${_rev_ks/1e9:.1f}B" if _rev_ks >= 1e9 else f"${_rev_ks/1e6:.0f}M" if _rev_ks else "N/A"

                _ks_items = [
                    ("P/E (TTM)",      f"{_pe_ks:.1f}x" if _pe_ks else "N/A",   "Trailing 12 months"),
                    ("Forward P/E",    f"{_fpe_ks:.1f}x" if _fpe_ks else "N/A", "NTM consensus est."),
                    ("EPS (TTM)",      f"${_eps_ks:.2f}" if _eps_ks else "N/A",  "Trailing 12 months"),
                    ("Revenue (TTM)",  _rev_str,                                  "Last 12 months"),
                    ("Dividend Yield", _div_str,                                  "Annual indicated"),
                    ("Beta (5Y)",      _beta_str,                                 "vs S&P 500"),
                ]
                _ks_cells = "".join(
                    f"<div style='flex:1;min-width:80px;padding:8px 6px;border-right:1px solid #21262d;'>"
                    f"<div style='font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px;'>{lbl}</div>"
                    f"<div style='font-size:15px;font-weight:700;color:#e6edf3;'>{val}</div>"
                    f"<div style='font-size:10px;color:#6e7681;margin-top:1px;'>{sub}</div></div>"
                    for lbl, val, sub in _ks_items
                )
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;overflow:hidden;">
  <div style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;font-weight:700;padding:8px 12px;background:#161b22;border-bottom:1px solid #30363d;">Key Statistics</div>
  <div style="display:flex;flex-wrap:wrap;">{_ks_cells}</div>
</div>""", unsafe_allow_html=True)

                # ── FINANCIAL QUALITY CARD ─────────────────────────────────────
                st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
                _gm_fq   = data.get('gross_margin', 0) or 0
                _opm_fq  = data.get('operating_margin', 0) or 0
                _npm_fq  = data.get('profit_margin', 0) or 0
                _fcf_fq  = data.get('free_cash_flow', 0) or 0
                _cash_fq = data.get('total_cash', 0) or 0
                _debt_fq = data.get('total_debt', 0) or 0
                _net_cash = _cash_fq - _debt_fq
                _net_cash_str = (f"${_net_cash/1e9:.1f}B" if abs(_net_cash) >= 1e9 else f"${_net_cash/1e6:.0f}M") if _net_cash != 0 else "N/A"
                _net_cash_label = "Net Cash" if _net_cash > 0 else "Net Debt"
                _fcf_str = (f"${_fcf_fq/1e9:.1f}B" if abs(_fcf_fq) >= 1e9 else f"${_fcf_fq/1e6:.0f}M") if _fcf_fq else "N/A"

                def _margin_color(pct, bench):
                    return "#3fb950" if pct >= bench else "#d29922" if pct >= bench * 0.5 else "#f85149"

                _fq_items = [
                    ("Gross Margin",  f"{_gm_fq*100:.1f}%",  "TTM", _margin_color(_gm_fq*100, 40)),
                    ("Op. Margin",    f"{_opm_fq*100:.1f}%", "TTM", _margin_color(_opm_fq*100, 15)),
                    ("Net Margin",    f"{_npm_fq*100:.1f}%", "TTM", _margin_color(_npm_fq*100, 10)),
                    ("FCF (TTM)",     _fcf_str,               "Free cash flow", "#3fb950" if _fcf_fq > 0 else "#f85149"),
                    (_net_cash_label, _net_cash_str,          "Cash minus debt", "#3fb950" if _net_cash > 0 else "#f85149"),
                ]
                _fq_cells = "".join(
                    f"<div style='flex:1;min-width:80px;padding:8px 6px;border-right:1px solid #21262d;'>"
                    f"<div style='font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px;'>{lbl}</div>"
                    f"<div style='font-size:15px;font-weight:700;color:{c};'>{val}</div>"
                    f"<div style='font-size:10px;color:#6e7681;margin-top:1px;'>{sub}</div></div>"
                    for lbl, val, sub, c in _fq_items
                )
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;overflow:hidden;">
  <div style="font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;font-weight:700;padding:8px 12px;background:#161b22;border-bottom:1px solid #30363d;">Financial Quality</div>
  <div style="display:flex;flex-wrap:wrap;">{_fq_cells}</div>
</div>""", unsafe_allow_html=True)


        # ── TECHNICAL TAB ──────────────────────────────────────────────────────
        with tab_tech:
            if not has_data:
                st.info("Enter a ticker and click **Run Full Analysis** to generate the report.")
            else:
                chart_data = tech_df.reset_index()
                chart_data['Date'] = pd.to_datetime(chart_data['Date']).dt.tz_convert(None)

                # ── 1. PROFESSIONAL OVERVIEW ─────────────────────────────────
                tech_score  = tech_analysis['score_pct']
                rsi_val     = tech_df['RSI'].iloc[-1]
                macd_val    = tech_df['MACD'].iloc[-1]
                macd_sig    = tech_df['MACD_Signal'].iloc[-1]
                sma50_val   = tech_df['SMA_50'].iloc[-1]
                sma200_val  = tech_df['SMA_200'].iloc[-1]
                close_val   = tech_df['Close'].iloc[-1]
                vs50  = (close_val / sma50_val  - 1) * 100 if sma50_val  and not pd.isna(sma50_val)  else 0
                vs200 = (close_val / sma200_val - 1) * 100 if sma200_val and not pd.isna(sma200_val) else 0

                if tech_score >= 30:
                    stance = "**bullish technical setup**"; s_color = "#3fb950"
                    outlook = "Price action shows sustained buying pressure. Trend indicators align positively, supporting continuation of the uptrend barring macro disruptions."
                elif tech_score >= 10:
                    stance = "**mildly constructive posture**"; s_color = "#3fb950"
                    outlook = "Positive signals outnumber negative ones, but conviction is limited. Watch for confirmation above key resistance to validate the bullish tilt."
                elif tech_score >= -10:
                    stance = "**mixed signals — no clear directional bias**"; s_color = "#d29922"
                    outlook = "Bullish and bearish indicators are roughly offsetting each other. No actionable directional edge; wait for a decisive break above resistance or below support."
                elif tech_score >= -30:
                    stance = "**mildly bearish deterioration**"; s_color = "#f85149"
                    outlook = "More indicators are pointing down than up. Exercise caution with new long positions; a confirmed bounce off support would be needed to turn the bias constructive."
                else:
                    stance = "**bearish technical breakdown**"; s_color = "#f85149"
                    outlook = "Selling pressure dominates price discovery across most indicators. Traders should exercise caution; wait for a confirmed technical base before re-entering long positions."

                rsi_desc  = (f"overbought at {rsi_val:.1f} — near-term exhaustion risk" if rsi_val > 70
                             else f"oversold at {rsi_val:.1f} — potential mean-reversion setup" if rsi_val < 30
                             else f"neutral at {rsi_val:.1f} — no directional bias from momentum")
                macd_desc = "MACD has crossed above its signal line, generating a bullish momentum signal" if macd_val > macd_sig else "MACD sits below its signal line, flagging negative momentum"
                sma_desc  = (f"{abs(vs50):.1f}% {'above' if vs50 >= 0 else 'below'} the 50-day SMA "
                             f"and {abs(vs200):.1f}% {'above' if vs200 >= 0 else 'below'} the 200-day SMA")

                st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px 22px;margin-bottom:6px;">
<p style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 6px 0;">Technical Analysis Overview</p>
<p style="color:#c9d1d9;line-height:1.8;margin:0;font-size:14px;">
<strong>{data['name']} ({data['ticker']})</strong> presents a {stance.replace("**","<strong>").replace("**","</strong>")}
 with a composite technical score of <strong style="color:{s_color};">{(tech_score+100)/2:.0f}/100</strong> (50 = neutral).
The stock is currently {sma_desc}, a key barometer of trend health — sustained trading above both moving averages 
signals an intact primary uptrend, while a cross below the 200-day SMA would indicate structural deterioration.
The RSI(14) reads {rsi_desc}.
{macd_desc}. {outlook}
</p></div>""", unsafe_allow_html=True)

                # ── 2. DIMENSION SCORE CARDS ─────────────────────────────────
                st.markdown("##### Dimensional Breakdown")
                bd = tech_analysis['breakdown']
                _show_overall = (tech_score + 100) / 2
                dim_cols = st.columns(5)
                dims = [
                    ("Overall",    _show_overall,            100, tech_analysis['rating'] + " · 50=neutral"),
                    ("Trend",      bd['trend']['score'],     bd['trend']['max'],      ""),
                    ("Momentum",   bd['momentum']['score'],  bd['momentum']['max'],   ""),
                    ("Volatility", bd['volatility']['score'],bd['volatility']['max'], ""),
                    ("Volume",     bd['volume']['score'],    bd['volume']['max'],     ""),
                ]
                for col, (label, score, mx, sub) in zip(dim_cols, dims):
                    pct = score / mx * 100 if mx else 0
                    bar_color = "#3fb950" if pct >= 65 else "#d29922" if pct >= 40 else "#f85149"
                    with col:
                        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px 10px;text-align:center;">
  <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;">{label}</div>
  <div style="font-size:22px;font-weight:700;color:{bar_color};margin:4px 0;">{score:.0f}<span style="font-size:13px;color:#6e7681;">/{mx}</span></div>
  <div style="background:#21262d;border-radius:4px;height:5px;margin:4px 0;">
    <div style="background:{bar_color};width:{pct:.0f}%;height:5px;border-radius:4px;"></div>
  </div>
  <div style="font-size:11px;color:#6e7681;">{sub if sub else f'{pct:.0f}%'}</div>
</div>""", unsafe_allow_html=True)

                # ── 3. PRICE + VOLUME CHART ───────────────────────────────────
                st.markdown("---")
                st.markdown("##### Price Action & Volume (1 Year)")
                st.markdown("""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:13px;color:#c9d1d9;line-height:1.7;">
<strong>How to read this chart:</strong> The price line (blue) represents daily closing prices over the past 12 months. 
The <span style="color:#f0883e"><strong>SMA 50</strong></span> (50-day Simple Moving Average) captures intermediate trend direction — 
a rising SMA 50 with price above it confirms a healthy uptrend. The <span style="color:#a371f7"><strong>SMA 200</strong></span> 
is the institutional benchmark for long-term trend health; a "Golden Cross" (SMA 50 crossing above SMA 200) is a classic 
bullish signal, while a "Death Cross" (SMA 50 crossing below SMA 200) flags long-term deterioration. 
<span style="color:#3fb950"><strong>Support</strong></span> and <span style="color:#f85149"><strong>Resistance</strong></span> 
levels mark price zones where supply/demand historically interacted. Volume bars at the bottom confirm price moves — 
high volume on up-days signals conviction buying; high volume on down-days signals distribution.
</div>""", unsafe_allow_html=True)

                _chart_type = st.radio("Chart type", ["Line", "Candlestick"], horizontal=True, key="chart_type_toggle")

                if _chart_type == "Candlestick":
                    import plotly.graph_objects as go
                    _hist_plot = data['hist_1y'].reset_index()
                    _hist_plot['Date'] = pd.to_datetime(_hist_plot['Date']).dt.tz_convert(None)
                    _fig_candle = go.Figure(data=[
                        go.Candlestick(
                            x=_hist_plot['Date'],
                            open=_hist_plot['Open'],
                            high=_hist_plot['High'],
                            low=_hist_plot['Low'],
                            close=_hist_plot['Close'],
                            name=data['ticker'],
                            increasing_line_color='#3fb950',
                            decreasing_line_color='#f85149',
                        )
                    ])
                    _fig_candle.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['SMA_50'].values,
                        mode='lines', name='SMA50', line=dict(color='#d29922', width=1.5, dash='dot')))
                    _fig_candle.add_trace(go.Scatter(x=chart_data['Date'], y=chart_data['SMA_200'].values,
                        mode='lines', name='SMA200', line=dict(color='#f85149', width=1.5, dash='dot')))
                    _fig_candle.update_layout(
                        paper_bgcolor='#0d1117', plot_bgcolor='#0d1117',
                        font=dict(color='#c9d1d9'), height=380,
                        xaxis=dict(gridcolor='#21262d', rangeslider_visible=False),
                        yaxis=dict(gridcolor='#21262d'),
                        legend=dict(bgcolor='rgba(0,0,0,0)'),
                        margin=dict(l=0, r=0, t=20, b=0),
                    )
                    st.plotly_chart(_fig_candle, use_container_width=True)
                    st.caption("Green = bullish candle  Red = bearish candle  Yellow = SMA50  Red dashed = SMA200")
                else:
                    price_layer = alt.Chart(chart_data).mark_line(color='#58a6ff', strokeWidth=2).encode(
                        x=alt.X('Date:T', title=None, axis=alt.Axis(format='%b %y', labelColor='#8b949e', gridColor='#21262d')),
                        y=alt.Y('Close:Q', title='Price (USD)', scale=alt.Scale(zero=False),
                                axis=alt.Axis(labelColor='#8b949e', gridColor='#21262d', format='$.0f')),
                        tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('Close:Q', format='$.2f', title='Close')]
                    )
                    ma50_layer  = alt.Chart(chart_data).mark_line(color='#f0883e', strokeWidth=1.5, strokeDash=[4,4]).encode(
                        x='Date:T', y=alt.Y('SMA_50:Q', title=''),
                        tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('SMA_50:Q', format='$.2f', title='SMA 50')]
                    )
                    ma200_layer = alt.Chart(chart_data).mark_line(color='#a371f7', strokeWidth=1.5, strokeDash=[4,4]).encode(
                        x='Date:T', y=alt.Y('SMA_200:Q', title=''),
                        tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('SMA_200:Q', format='$.2f', title='SMA 200')]
                    )
                    price_combined = price_layer + ma50_layer + ma200_layer
                    if supports:
                        sup_df = pd.DataFrame({'y': [supports[-1]]})
                        price_combined = price_combined + alt.Chart(sup_df).mark_rule(color='#3fb950', strokeWidth=1.5, strokeDash=[3,3]).encode(y='y:Q')
                    if resistances:
                        res_df = pd.DataFrame({'y': [resistances[-1]]})
                        price_combined = price_combined + alt.Chart(res_df).mark_rule(color='#f85149', strokeWidth=1.5, strokeDash=[3,3]).encode(y='y:Q')

                    vol_layer = alt.Chart(chart_data).mark_bar(opacity=0.6).encode(
                        x=alt.X('Date:T', title=None),
                        y=alt.Y('Volume:Q', title='Volume', axis=alt.Axis(labelColor='#8b949e', format='~s', gridColor='#21262d')),
                        color=alt.condition(
                            alt.datum.Close >= alt.datum.Open if 'Open' in chart_data.columns else alt.value('#3fb950'),
                            alt.value('#3fb950'), alt.value('#f85149')
                        ),
                        tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('Volume:Q', format=',', title='Volume')]
                    )

                    combined_chart = alt.vconcat(
                        price_combined.properties(height=280, title=''),
                        vol_layer.properties(height=80, title=''),
                    ).resolve_scale(x='shared').configure_view(strokeWidth=0).configure(background='#0d1117').configure_axis(
                        labelFontSize=11, titleFontSize=11, titleColor='#8b949e', domainColor='#30363d'
                    )
                    st.altair_chart(combined_chart, use_container_width=True)
                    st.caption("🔵 Price  |  🟠 SMA 50  |  🟣 SMA 200  |  🟢 Support  |  🔴 Resistance  |  Bars: Volume")

                # ── 4. RSI SECTION ────────────────────────────────────────────
                st.markdown("---")
                st.markdown("##### Momentum Oscillator: RSI (14-Period)")
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:13px;color:#c9d1d9;line-height:1.7;">
The <strong>Relative Strength Index (RSI)</strong>, developed by J. Welles Wilder, is a momentum oscillator ranging 0–100 
that measures the speed and magnitude of recent price changes. It compares average gains to average losses over a 14-period 
rolling window. <strong style="color:#f85149;">RSI &gt; 70</strong> signals overbought conditions — the asset may be due for a 
consolidation or reversal. <strong style="color:#3fb950;">RSI &lt; 30</strong> signals oversold conditions — historically a 
mean-reversion opportunity. The 50 midline acts as a trend filter: sustained readings above 50 confirm bullish momentum, 
below 50 confirm bearish momentum. Currently, {data['ticker']}'s RSI(14) is 
<strong style="color={'#f85149' if rsi_val > 70 else '#3fb950' if rsi_val < 30 else '#d29922'};">{rsi_val:.1f}</strong> — {rsi_desc}.
</div>""", unsafe_allow_html=True)

                rsi_data = chart_data.tail(120).copy()
                rsi_line = alt.Chart(rsi_data).mark_line(color='#58a6ff', strokeWidth=2).encode(
                    x=alt.X('Date:T', title=None, axis=alt.Axis(format='%b %y', labelColor='#8b949e', gridColor='#21262d')),
                    y=alt.Y('RSI:Q', scale=alt.Scale(domain=[0, 100]), title='RSI',
                            axis=alt.Axis(labelColor='#8b949e', gridColor='#21262d', values=[20,30,50,70,80])),
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('RSI:Q', format='.1f', title='RSI')]
                )
                ob_rule = alt.Chart(pd.DataFrame({'y': [70]})).mark_rule(color='#f85149', strokeWidth=1.5, strokeDash=[4,4]).encode(y='y:Q')
                os_rule = alt.Chart(pd.DataFrame({'y': [30]})).mark_rule(color='#3fb950', strokeWidth=1.5, strokeDash=[4,4]).encode(y='y:Q')
                mid_rule = alt.Chart(pd.DataFrame({'y': [50]})).mark_rule(color='#6e7681', strokeWidth=1, strokeDash=[2,4]).encode(y='y:Q')
                rsi_chart = (rsi_line + ob_rule + os_rule + mid_rule).properties(height=200).configure_view(strokeWidth=0).configure(background='#0d1117').configure_axis(labelFontSize=11, titleFontSize=11, titleColor='#8b949e', domainColor='#30363d')
                st.altair_chart(rsi_chart, use_container_width=True)
                rsi_box_color = "#f85149" if rsi_val > 70 else "#3fb950" if rsi_val < 30 else "#d29922"
                st.markdown(f"<div style='background:{rsi_box_color}22;border:1px solid {rsi_box_color};border-radius:6px;padding:8px 14px;font-size:13px;color:#c9d1d9;'>RSI(14): <strong style='color:{rsi_box_color};'>{rsi_val:.1f}</strong> — {'🔴 Overbought — consider tightening stops or reducing exposure' if rsi_val > 70 else '🟢 Oversold — watch for reversal confirmation before entering' if rsi_val < 30 else '🟡 Neutral — momentum provides no strong directional edge'}</div>", unsafe_allow_html=True)

                # ── 5. MACD SECTION ───────────────────────────────────────────
                st.markdown("---")
                st.markdown("##### Trend Momentum: MACD (12, 26, 9)")
                macd_diff   = macd_val - macd_sig
                macd_trend  = "expanding" if abs(chart_data['MACD_Hist'].iloc[-1]) > abs(chart_data['MACD_Hist'].iloc[-5]) else "contracting"
                if macd_diff > 0 and macd_trend == 'expanding':
                    _macd_momentum_desc = "building bullish momentum"
                elif macd_diff > 0 and macd_trend == 'contracting':
                    _macd_momentum_desc = "fading bullish momentum — upward pressure is easing"
                elif macd_diff < 0 and macd_trend == 'expanding':
                    _macd_momentum_desc = "intensifying bearish pressure"
                else:
                    _macd_momentum_desc = "easing bearish pressure — downward momentum is waning"
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:13px;color:#c9d1d9;line-height:1.7;">
The <strong>MACD (Moving Average Convergence Divergence)</strong> indicator, created by Gerald Appel, measures
the relationship between two exponential moving averages — the 12-period EMA minus the 26-period EMA — to identify
trend direction and momentum shifts. The <strong>Signal Line</strong> (9-period EMA of MACD) acts as a trigger:
when the MACD line crosses above it, a bullish signal is generated; a cross below is bearish. The
<strong>Histogram</strong> (the difference between MACD and Signal) provides a visual representation of momentum
— widening bars in the direction of the trend indicate strengthening conviction, narrowing bars signal fading pressure.
Currently, MACD is <strong style="color={'#58a6ff'};">{macd_val:.3f}</strong> vs Signal <strong>{macd_sig:.3f}</strong>
(spread: <strong style="color={'#3fb950' if macd_diff >= 0 else '#f85149'};">{macd_diff:+.3f}</strong>),
with the histogram <strong>{macd_trend}</strong> — suggesting <strong>{_macd_momentum_desc}</strong>.
</div>""", unsafe_allow_html=True)

                macd_data  = chart_data.tail(120).copy()
                macd_line  = alt.Chart(macd_data).mark_line(color='#58a6ff', strokeWidth=2).encode(
                    x=alt.X('Date:T', title=None, axis=alt.Axis(format='%b %y', labelColor='#8b949e', gridColor='#21262d')),
                    y=alt.Y('MACD:Q', title='MACD', axis=alt.Axis(labelColor='#8b949e', gridColor='#21262d')),
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('MACD:Q', format='.3f', title='MACD')]
                )
                sig_line   = alt.Chart(macd_data).mark_line(color='#f0883e', strokeWidth=1.5, strokeDash=[4,4]).encode(
                    x='Date:T', y=alt.Y('MACD_Signal:Q', title=''),
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('MACD_Signal:Q', format='.3f', title='Signal')]
                )
                hist_bars  = alt.Chart(macd_data).mark_bar(opacity=0.65).encode(
                    x='Date:T',
                    y=alt.Y('MACD_Hist:Q', title=''),
                    color=alt.condition(alt.datum.MACD_Hist >= 0, alt.value('#3fb950'), alt.value('#f85149')),
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('MACD_Hist:Q', format='.3f', title='Histogram')]
                )
                zero_rule  = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(color='#6e7681', strokeWidth=1).encode(y='y:Q')
                macd_chart = (hist_bars + macd_line + sig_line + zero_rule).properties(height=220).configure_view(strokeWidth=0).configure(background='#0d1117').configure_axis(labelFontSize=11, titleFontSize=11, titleColor='#8b949e', domainColor='#30363d')
                st.altair_chart(macd_chart, use_container_width=True)
                st.caption("🔵 MACD Line  |  🟠 Signal Line  |  🟢/🔴 Histogram (momentum bars)")

                # ── 6. SIGNAL DETAIL TABLE ────────────────────────────────────
                st.markdown("---")
                st.markdown("##### Full Signal Detail")
                tech_signals_df = pd.DataFrame(tech_analysis['signals'])
                if not tech_signals_df.empty:
                    _rows = ""
                    for _, _r in tech_signals_df.iterrows():
                        _s = str(_r.get('signal', '')).upper()
                        _fg = "#3fb950" if any(x in _s for x in ['BULLISH','BUY','STRONG','ABOVE']) else "#f85149" if any(x in _s for x in ['BEARISH','SELL','WEAK','BELOW']) else "#d29922"
                        _bg = _fg + "18"
                        _rows += (f"<tr style='border-bottom:1px solid #21262d;'>"
                            f"<td style='padding:8px 10px;font-size:11px;color:#8b949e;text-transform:uppercase;white-space:nowrap;'>{_r.get('category','')}</td>"
                            f"<td style='padding:8px 10px;font-size:13px;color:#e6edf3;font-weight:500;'>{_r.get('indicator','')}</td>"
                            f"<td style='padding:8px 10px;'><span style='background:#0d1117;border:1px solid {_fg};color:{_fg} !important;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;'>{_r.get('signal','')}</span></td>"
                            f"<td style='padding:8px 10px;font-size:14px;color:{_fg};font-weight:700;text-align:center;'>{_r.get('score','')}</td>"
                            f"<td style='padding:8px 10px;font-size:12px;color:#8b949e;'>{_r.get('detail','')}</td>"
                            f"<td style='padding:8px 10px;font-size:11px;color:#6e7681;text-align:center;'>{_r.get('threshold','')}</td>"
                            f"</tr>")
                    st.markdown(
                        "<div style='background:#0d1117;border:1px solid #30363d;border-radius:8px;overflow:hidden;'>"
                        "<table style='width:100%;border-collapse:collapse;'>"
                        "<thead><tr style='background:#161b22;border-bottom:2px solid #30363d;'>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:left;text-transform:uppercase;'>Category</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:left;text-transform:uppercase;'>Indicator</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:left;text-transform:uppercase;'>Signal</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:center;text-transform:uppercase;'>Score</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:left;text-transform:uppercase;'>Detail</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:center;text-transform:uppercase;'>Threshold</th>"
                        f"</tr></thead><tbody>{_rows}</tbody></table></div>",
                        unsafe_allow_html=True)
                st.caption("Data: Yahoo Finance real-time feed (15\u201320 min delayed). Scores are proprietary CFA-style composite ratings.")

        # ── FUNDAMENTAL TAB ────────────────────────────────────────────────────
        with tab_fund:
            if not has_data:
                st.info("Run an analysis to see fundamental data.")
            else:
                # ── 1. PROFESSIONAL OVERVIEW ─────────────────────────────────
                fund_score = fund_analysis.get('display_score', max(0.0, min(100.0, 50.0 + fund_analysis['score_pct'] * 0.5)))
                pe    = data.get('pe_ratio')
                fpe   = data.get('forward_pe')
                # PEG: use yfinance value with fallback calculation (single source)
                peg   = data.get('peg_ratio')
                if not peg:
                    _pe_raw = data.get('pe_ratio')
                    _eg_raw = data.get('earnings_growth', 0) or 0
                    if _pe_raw and _eg_raw > 0:
                        peg = _pe_raw / (_eg_raw * 100)
                roe   = data.get('roe', 0) or 0
                npm   = data.get('profit_margin', 0) or 0
                opm   = data.get('operating_margin', 0) or 0
                rev_g = data.get('revenue_growth', 0) or 0
                eps_g = data.get('earnings_growth', 0) or 0
                de    = data.get('debt_to_equity', 0) or 0
                cr    = data.get('current_ratio', 0) or 0
                fcf   = data.get('free_cash_flow', 0) or 0

                if fund_score >= 70:
                    f_stance = "**fundamentally strong**"; f_color = "#3fb950"
                    f_outlook = "The business demonstrates durable competitive advantages with solid earnings quality, manageable leverage, and a growth trajectory that justifies current valuation."
                elif fund_score >= 55:
                    f_stance = "**fundamentally sound**"; f_color = "#3fb950"
                    f_outlook = "Core financials are solid with most metrics in acceptable ranges. The business generates adequate returns on capital, though some areas warrant monitoring."
                elif fund_score >= 40:
                    f_stance = "**fundamentally mixed**"; f_color = "#d29922"
                    f_outlook = "Strengths and weaknesses coexist. Some financial dimensions are adequate while others lag benchmarks — monitor for improving or deteriorating trends."
                else:
                    f_stance = "**fundamentally challenged**"; f_color = "#f85149"
                    f_outlook = "Financial metrics lag sector benchmarks across multiple dimensions. Thorough due diligence is essential before establishing a position."

                val_comment = (f"trading at {pe:.1f}x trailing earnings" if pe else "P/E not available")
                roe_comment = (f"return on equity of {roe*100:.1f}%" if roe else "ROE not available")
                fcf_comment = (f"free cash flow of ${fcf/1e9:.1f}B — a direct measure of capital generation capacity" if fcf else "FCF not disclosed")

                st.markdown(f"""
<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:20px 22px;margin-bottom:6px;">
<p style="color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:1px;margin:0 0 6px 0;">Fundamental Analysis Overview</p>
<p style="color:#c9d1d9;line-height:1.8;margin:0;font-size:14px;">
<strong>{data['name']} ({data['ticker']})</strong> is assessed as
<strong style="color:{f_color};">{f_stance.replace('**','')}</strong>
 with a composite fundamental score of <strong style="color:{f_color};">{fund_score:.0f}/100</strong> (50 = neutral)
across valuation, profitability, growth, and financial health dimensions.
The stock is currently {val_comment},
posting a {roe_comment}, and generating {fcf_comment}.
{f_outlook}
</p></div>""", unsafe_allow_html=True)

                # ── 2. DIMENSION SCORE CARDS ─────────────────────────────────
                st.markdown("##### Dimensional Breakdown")
                bd = fund_analysis['breakdown']
                dim_cols = st.columns(5)
                dims = [
                    ("Overall",       fund_score,                      100, fund_analysis['rating']),
                    ("Valuation",     bd['valuation']['score'],         bd['valuation']['max'],     ""),
                    ("Profitability", bd['profitability']['score'],     bd['profitability']['max'], ""),
                    ("Growth",        bd['growth']['score'],            bd['growth']['max'],        ""),
                    ("Health",        bd['health']['score'],            bd['health']['max'],        ""),
                ]
                for col, (label, score, mx, sub) in zip(dim_cols, dims):
                    pct = score / mx * 100 if mx else 0
                    bar_color = "#3fb950" if pct >= 65 else "#d29922" if pct >= 40 else "#f85149"
                    with col:
                        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px 10px;text-align:center;">
  <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;">{label}</div>
  <div style="font-size:22px;font-weight:700;color:{bar_color};margin:4px 0;">{score:.0f}<span style="font-size:13px;color:#6e7681;">/{mx}</span></div>
  <div style="background:#21262d;border-radius:4px;height:5px;margin:4px 0;">
    <div style="background:{bar_color};width:{pct:.0f}%;height:5px;border-radius:4px;"></div>
  </div>
  <div style="font-size:11px;color:#6e7681;">{sub if sub else f'{pct:.0f}%'}</div>
</div>""", unsafe_allow_html=True)

                # ── 3. VALUATION ──────────────────────────────────────────────
                st.markdown("---")
                st.markdown("##### Valuation Multiples")
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:13px;color:#c9d1d9;line-height:1.7;">
<strong>Valuation multiples</strong> benchmark a stock's market price against its fundamental earnings and cash flow 
metrics to assess whether it trades at a premium or discount. The <strong>Price-to-Earnings (P/E)</strong> ratio divides 
market price by trailing twelve-month EPS — historically, S&P 500 averages ~16–18x. <strong>Forward P/E</strong> uses 
consensus EPS estimates, reflecting market expectations for earnings growth. The 
<strong>PEG Ratio</strong> (P/E divided by earnings growth rate) adjusts valuation for growth; values below 1.0x 
suggest undervaluation relative to growth. <strong>EV/EBITDA</strong> is an enterprise-level metric preferred by 
credit analysts and M&A practitioners for cross-capital-structure comparisons — sub-10x typically indicates 
relative value.{f" Currently, {data['ticker']} trades at {pe:.1f}x trailing P/E and {fpe:.1f}x forward P/E, implying {'earnings growth expectations are priced in' if fpe and pe and fpe < pe else 'the market anticipates margin compression or earnings decline' if fpe and pe and fpe > pe else ''}." if pe else ""}
</div>""", unsafe_allow_html=True)

                v_col1, v_col2, v_col3, v_col4 = st.columns(4)
                metrics_val = [
                    ("P/E Ratio", f"{pe:.1f}x" if pe else "N/A", "Trailing TTM", 18, pe),
                    ("Forward P/E", f"{fpe:.1f}x" if fpe else "N/A", "Next 12M est.", 16, fpe),
                    ("PEG Ratio", f"{peg:.2f}" if peg else "N/A", "<1.0 = value", 1, peg),
                    ("EV/EBITDA", f"{data.get('ev_ebitda',0):.1f}x" if data.get('ev_ebitda') else "N/A", "Enterprise value", 12, data.get('ev_ebitda')),
                ]
                for col, (label, val, sub, bench, raw) in zip([v_col1,v_col2,v_col3,v_col4], metrics_val):
                    if raw:
                        is_high = (raw > bench * 1.5) if label != "PEG Ratio" else (raw > 1.5)
                        is_ok   = (raw > bench * 0.8) if label != "PEG Ratio" else (raw > 0.8)
                        dot_c   = "#f85149" if is_high else "#3fb950" if not is_ok else "#d29922"
                    else:
                        dot_c = "#6e7681"
                    with col:
                        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px 10px;text-align:center;">
  <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;">{label}</div>
  <div style="font-size:20px;font-weight:700;color:{dot_c};margin:6px 0;">{val}</div>
  <div style="font-size:11px;color:#6e7681;">{sub}</div>
</div>""", unsafe_allow_html=True)

                # ── 4. PROFITABILITY ──────────────────────────────────────────
                st.markdown("---")
                st.markdown("##### Profitability & Returns")
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:13px;color:#c9d1d9;line-height:1.7;">
<strong>Profitability metrics</strong> quantify management's efficiency in converting revenue into earnings and the 
quality of returns generated on invested capital. <strong>Operating Margin</strong> measures core business profitability 
before interest and taxes — consistently expanding margins signal pricing power and operating leverage. 
<strong>Net Profit Margin</strong> captures the bottom-line percentage retained after all costs. 
<strong>Return on Equity (ROE)</strong> is the primary measure of shareholder capital efficiency; Buffett's benchmark 
is consistently above 15%. <strong>Return on Assets (ROA)</strong> measures asset utilisation efficiency — 
particularly relevant for capital-intensive industries. {data['ticker']} posts an operating margin of 
<strong>{opm*100:.1f}%</strong>, net margin of <strong>{npm*100:.1f}%</strong>, and ROE of 
<strong>{roe*100:.1f}%</strong>.
</div>""", unsafe_allow_html=True)

                p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                prof_metrics = [
                    ("Op. Margin",  f"{opm*100:.1f}%",  "≥15% strong",  opm*100,  15),
                    ("Net Margin",  f"{npm*100:.1f}%",  "≥10% strong",  npm*100,  10),
                    ("ROE",         f"{roe*100:.1f}%",  "≥15% Buffett", roe*100,  15),
                    ("ROA",         f"{(data.get('roa',0) or 0)*100:.1f}%", "≥5% solid", (data.get('roa',0) or 0)*100, 5),
                ]
                for col, (label, val, sub, raw_pct, bench) in zip([p_col1,p_col2,p_col3,p_col4], prof_metrics):
                    c = "#3fb950" if raw_pct >= bench else "#d29922" if raw_pct >= bench * 0.5 else "#f85149"
                    with col:
                        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px 10px;text-align:center;">
  <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;">{label}</div>
  <div style="font-size:20px;font-weight:700;color:{c};margin:6px 0;">{val}</div>
  <div style="font-size:11px;color:#6e7681;">{sub}</div>
</div>""", unsafe_allow_html=True)

                # ── 5. GROWTH ─────────────────────────────────────────────────
                st.markdown("---")
                st.markdown("##### Growth Trajectory")
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:13px;color:#c9d1d9;line-height:1.7;">
<strong>Growth metrics</strong> are the primary drivers of intrinsic value in DCF models and determine whether a 
premium valuation multiple is justified. <strong>Revenue Growth</strong> reflects market share gains and pricing power; 
secular growth companies consistently compound revenues at 15–25%+ annually. <strong>EPS Growth</strong> tracks 
earnings power expansion — companies growing EPS faster than revenue are expanding margins or buying back shares 
aggressively. Forward EPS estimates reveal consensus expectations, acting as the anchor for price target models. 
{data['ticker']} is growing revenue at <strong>{rev_g*100:.1f}%</strong> YoY with EPS growth of 
<strong>{eps_g*100:.1f}%</strong>, implying {'earnings growing faster than revenues — margin expansion or buybacks at work' if eps_g > rev_g else 'margin dilution or elevated opex outpacing revenue growth'}.
</div>""", unsafe_allow_html=True)

                g_col1, g_col2, g_col3, g_col4 = st.columns(4)
                eps_val = data.get('eps', 0) or 0
                feps_val = data.get('forward_eps', 0) or 0
                growth_metrics = [
                    ("Revenue Growth", f"{rev_g*100:.1f}%", "YoY",        rev_g*100,  10),
                    ("EPS Growth",     f"{eps_g*100:.1f}%", "YoY",        eps_g*100,  10),
                    ("EPS (TTM)",      f"${eps_val:.2f}",   "Trailing",   eps_val,    0),
                    ("Fwd EPS",        f"${feps_val:.2f}",  "Consensus",  feps_val,   0),
                ]
                for col, (label, val, sub, raw, bench) in zip([g_col1,g_col2,g_col3,g_col4], growth_metrics):
                    c = "#3fb950" if raw > bench else "#d29922" if raw > 0 else "#f85149"
                    if label in ("EPS (TTM)", "Fwd EPS"):
                        c = "#3fb950" if raw > 0 else "#f85149"
                    with col:
                        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px 10px;text-align:center;">
  <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;">{label}</div>
  <div style="font-size:20px;font-weight:700;color:{c};margin:6px 0;">{val}</div>
  <div style="font-size:11px;color:#6e7681;">{sub}</div>
</div>""", unsafe_allow_html=True)

                # ── 6. FINANCIAL HEALTH ───────────────────────────────────────
                st.markdown("---")
                st.markdown("##### Balance Sheet & Financial Health")
                cash_val = data.get('total_cash', 0) or 0
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:13px;color:#c9d1d9;line-height:1.7;">
<strong>Financial health metrics</strong> assess balance sheet resilience, liquidity, and capital structure risk. 
<strong>Debt/Equity (D/E)</strong> measures financial leverage — ratios above 2.0x raise solvency concerns in 
rising-rate environments, though capital-light businesses (software, platforms) often carry more debt responsibly. 
<strong>Current Ratio</strong> (current assets ÷ current liabilities) is the primary short-term liquidity gauge; 
values above 1.5x indicate a comfortable buffer, below 1.0x signals potential near-term cash pressure. 
<strong>Free Cash Flow (FCF)</strong> — operating cash flow minus capex — is Warren Buffett's preferred measure of 
intrinsic earnings quality; FCF-generative businesses can self-fund growth, pay dividends, and execute buybacks 
without diluting shareholders. {data['ticker']} carries a D/E of <strong>{de:.2f}x</strong>, 
current ratio of <strong>{cr:.2f}x</strong>, and generates <strong>{'${:,.1f}B'.format(fcf/1e9) if fcf else 'N/A'}</strong> in FCF.
</div>""", unsafe_allow_html=True)

                h_col1, h_col2, h_col3, h_col4 = st.columns(4)
                health_metrics = [
                    ("Debt / Equity",   f"{de:.2f}x",              "≤2.0 safe",    de,       2.0,  True),
                    ("Current Ratio",   f"{cr:.2f}x",              "≥1.5 liquid",  cr,       1.5,  False),
                    ("Free Cash Flow",  f"${fcf/1e9:.1f}B" if fcf else "N/A", "Cash generation", fcf/1e9 if fcf else 0, 0, False),
                    ("Total Cash",      f"${cash_val/1e9:.1f}B" if cash_val else "N/A", "Liquidity buffer", cash_val/1e9 if cash_val else 0, 0, False),
                ]
                for col, (label, val, sub, raw, bench, invert) in zip([h_col1,h_col2,h_col3,h_col4], health_metrics):
                    if invert:
                        c = "#f85149" if raw > bench * 1.5 else "#d29922" if raw > bench else "#3fb950"
                    else:
                        c = "#3fb950" if raw >= bench else "#d29922" if raw > 0 else "#f85149"
                    with col:
                        st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:12px 10px;text-align:center;">
  <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:.5px;">{label}</div>
  <div style="font-size:20px;font-weight:700;color:{c};margin:6px 0;">{val}</div>
  <div style="font-size:11px;color:#6e7681;">{sub}</div>
</div>""", unsafe_allow_html=True)

                # ── 7. VALUATION MODEL COMPARISON CHART ──────────────────────
                if valuation:
                    st.markdown("---")
                    st.markdown("##### Valuation Model Price Targets")
                    st.markdown("""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:13px;color:#c9d1d9;line-height:1.7;">
The chart below plots bear / base / bull price targets from four independent valuation models alongside the 
current market price. <strong>P/E Based</strong> uses historical earnings multiples applied to trailing EPS. 
<strong>Forward P/E</strong> applies consensus future earnings estimates. <strong>Analyst Consensus</strong> 
aggregates price targets from major sell-side brokerages. <strong>DCF</strong> (Discounted Cash Flow) models 
intrinsic value based on projected free cash flows discounted at WACC — the most theoretically rigorous but 
also most assumption-dependent approach. A convergence of models above the current price signals a margin of safety.
</div>""", unsafe_allow_html=True)

                    val_rows = []
                    current_price = data['price']
                    model_map = [
                        ('pe_valuation',      'P/E Based'),
                        ('forward_pe_valuation','Forward P/E'),
                        ('analyst_target',    'Analyst Consensus'),
                        ('dcf_valuation',     'DCF Model'),
                    ]
                    for key, model_name in model_map:
                        if key in valuation:
                            v = valuation[key]
                            low  = v.get('low')  or v.get('bear')
                            mid  = v.get('mid')  or v.get('base')
                            high = v.get('high') or v.get('bull')
                            if low  is not None: val_rows.append({'Model': model_name, 'Scenario': 'Bear', 'Price': low})
                            if mid  is not None: val_rows.append({'Model': model_name, 'Scenario': 'Base', 'Price': mid})
                            if high is not None: val_rows.append({'Model': model_name, 'Scenario': 'Bull', 'Price': high})

                    if val_rows:
                        val_df = pd.DataFrame(val_rows)
                        color_scale = alt.Scale(domain=['Bear','Base','Bull'], range=['#f85149','#d29922','#3fb950'])
                        bars = alt.Chart(val_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                            x=alt.X('Model:N', title=None, axis=alt.Axis(labelColor='#8b949e', labelAngle=0)),
                            y=alt.Y('Price:Q', title='Price (USD)', scale=alt.Scale(zero=False),
                                    axis=alt.Axis(labelColor='#8b949e', gridColor='#21262d', format='$.0f')),
                            color=alt.Color('Scenario:N', scale=color_scale, legend=alt.Legend(orient='top', labelColor='#c9d1d9', titleColor='#8b949e')),
                            xOffset='Scenario:N',
                            tooltip=[alt.Tooltip('Model:N'), alt.Tooltip('Scenario:N'), alt.Tooltip('Price:Q', format='$.2f')]
                        )
                        price_line = alt.Chart(pd.DataFrame({'y': [current_price]})).mark_rule(color='#58a6ff', strokeWidth=2, strokeDash=[5,4]).encode(y='y:Q')
                        val_chart = (bars + price_line).properties(height=260).configure_view(strokeWidth=0).configure(background='#0d1117').configure_axis(labelFontSize=11, titleFontSize=11, titleColor='#8b949e', domainColor='#30363d')
                        st.altair_chart(val_chart, use_container_width=True)
                        st.caption(f"🔵 Dashed line = current price (${current_price:.2f})  |  🔴 Bear  |  🟡 Base  |  🟢 Bull")

                    if 'composite' in valuation:
                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Composite Bear", f"${valuation['composite']['target_low']:.2f}", f"{valuation['composite']['upside_low']:+.1f}%")
                        with col2:
                            st.metric("Composite Base", f"${valuation['composite']['target_mid']:.2f}", f"{valuation['composite']['upside_mid']:+.1f}%")
                        with col3:
                            st.metric("Composite Bull", f"${valuation['composite']['target_high']:.2f}", f"{valuation['composite']['upside_high']:+.1f}%")

                # ── 8. SIGNAL DETAIL TABLE ────────────────────────────────────
                st.markdown("---")
                st.markdown("##### Full Signal Detail")
                fund_signals_df = pd.DataFrame(fund_analysis['signals'])
                if not fund_signals_df.empty:
                    _rows = ""
                    for _, _r in fund_signals_df.iterrows():
                        _s = str(_r.get('signal', '')).upper()
                        _fg = "#3fb950" if any(x in _s for x in ['BULLISH','BUY','STRONG','POSITIVE','HEALTHY','GOOD','HIGH']) else "#f85149" if any(x in _s for x in ['BEARISH','SELL','WEAK','NEGATIVE','POOR','RISKY']) else "#d29922"
                        _bg = _fg + "18"
                        _rows += (f"<tr style='border-bottom:1px solid #21262d;'>"
                            f"<td style='padding:8px 10px;font-size:11px;color:#8b949e;text-transform:uppercase;white-space:nowrap;'>{_r.get('category','')}</td>"
                            f"<td style='padding:8px 10px;font-size:13px;color:#e6edf3;font-weight:500;'>{_r.get('metric','')}</td>"
                            f"<td style='padding:8px 10px;font-size:13px;color:#58a6ff;font-weight:600;text-align:right;white-space:nowrap;'>{_r.get('value','')}</td>"
                            f"<td style='padding:8px 10px;'><span style='background:#0d1117;border:1px solid {_fg};color:{_fg} !important;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;'>{_r.get('signal','')}</span></td>"
                            f"<td style='padding:8px 10px;font-size:14px;color:{_fg};font-weight:700;text-align:center;'>{_r.get('score','')}</td>"
                            f"<td style='padding:8px 10px;font-size:12px;color:#8b949e;'>{_r.get('detail','')}</td>"
                            f"<td style='padding:8px 10px;font-size:11px;color:#6e7681;text-align:center;white-space:nowrap;'>{_r.get('benchmark','')}</td>"
                            f"</tr>")
                    st.markdown(
                        "<div style='background:#0d1117;border:1px solid #30363d;border-radius:8px;overflow:hidden;'>"
                        "<table style='width:100%;border-collapse:collapse;'>"
                        "<thead><tr style='background:#161b22;border-bottom:2px solid #30363d;'>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:left;text-transform:uppercase;'>Category</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:left;text-transform:uppercase;'>Metric</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:right;text-transform:uppercase;'>Value</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:left;text-transform:uppercase;'>Signal</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:center;text-transform:uppercase;'>Score</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:left;text-transform:uppercase;'>Detail</th>"
                        "<th style='padding:10px;font-size:11px;color:#8b949e;text-align:center;text-transform:uppercase;'>Benchmark</th>"
                        f"</tr></thead><tbody>{_rows}</tbody></table></div>",
                        unsafe_allow_html=True)
                st.caption("Source: Yahoo Finance \u2014 SEC filings (10-K, 10-Q). Analyst estimates aggregated from major sell-side brokerages.")

                st.markdown("---")
                st.markdown("**Peer Comparison**")
                st.caption("Key metrics vs sector peers (Yahoo Finance data)")
                with st.spinner("Loading peers..."):
                    _peer_df = fetch_peer_comparison(data['ticker'], data.get('sector', ''))
                if not _peer_df.empty:
                    def _style_peers(row):
                        if _peer_df.loc[row.name, '_is_subject']:
                            return ['background-color: rgba(29,78,216,0.15); font-weight:bold'] * len(row)
                        return [''] * len(row)
                    _display_df = _peer_df.drop(columns=['_is_subject'])
                    st.dataframe(_display_df.style.apply(_style_peers, axis=1),
                                 use_container_width=True, hide_index=True)

        # ── CONCLUSION TAB ─────────────────────────────────────────────────────
        with tab_conclusion:
            if not has_data:
                st.info("Run an analysis to see the conclusion and forecast.")
            else:
                ac = recommendation['action_color']
                upside_low  = (recommendation['target_low']  - data['price']) / data['price'] * 100
                upside_high = (recommendation['target_high'] - data['price']) / data['price'] * 100

                # Sanity check 1: never show negative "upside"
                _up = recommendation['upside']
                _upside_label = (
                    f"{abs(_up):.1f}% potential downside"
                    if _up < 0
                    else f"{_up:.1f}% potential upside"
                )

                # Sanity check 2: rename Bull → Best when bull target ≤ current price
                _bull_label = "Bull Case" if recommendation['target_high'] > data['price'] else "Best Case"
                _bull_note  = "" if recommendation['target_high'] > data['price'] else "<div style='font-size:10px;color:#8b949e;margin-top:4px;'>best case is still downside</div>"

                # ── 1. HEADLINE RECOMMENDATION CARD ──────────────────────────
                st.markdown(f"""
<div style="background:#0d1117;border:2px solid {ac};border-radius:14px;padding:28px 30px;margin-bottom:18px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:16px;">
    <div>
      <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Investment Rating</div>
      <div style="font-size:48px;font-weight:800;color:{ac};letter-spacing:-1px;line-height:1;">{recommendation['action']}</div>
      <div style="font-size:15px;color:#c9d1d9;margin-top:10px;">{recommendation['trade_decision']}</div>
      <div style="margin-top:14px;">
        <span style="background:#0d1117;border:1px solid {ac};border-radius:20px;padding:5px 16px;font-size:12px;color:{ac};font-weight:600;">
          Composite Score: {recommendation['combined_score']:.0f} / 100 (50 = neutral)
        </span>
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:11px;color:#8b949e;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;">12-Month Price Target</div>
      <div style="font-size:40px;font-weight:700;color:#e6edf3;line-height:1;">${recommendation['target_price']:.2f}</div>
      <div style="font-size:17px;color:{ac};font-weight:600;margin-top:4px;">{_upside_label}</div>
      <div style="font-size:12px;color:#6e7681;margin-top:8px;">Current price: ${data['price']:.2f}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

                # ── 2. PRICE TARGET SCENARIOS ─────────────────────────────────
                st.markdown("##### Price Target Scenarios")
                pt_col1, pt_col2, pt_col3 = st.columns(3)
                for col, (label, icon, price, upside, color, note) in zip(
                    [pt_col1, pt_col2, pt_col3],
                    [
                        ("Bear Case",  "\U0001f43b", recommendation['target_low'],   upside_low,   "#f85149", ""),
                        ("Base Case",  "\u2696\ufe0f",  recommendation['target_price'], recommendation['upside'], "#d29922", ""),
                        (_bull_label,  "\U0001f402", recommendation['target_high'],  upside_high,  "#3fb950", _bull_note),
                    ]
                ):
                    with col:
                        st.markdown(f"""
<div style="background:#0d1117;border:1px solid {color};border-radius:10px;padding:18px 14px;text-align:center;">
  <div style="font-size:22px;margin-bottom:4px;">{icon}</div>
  <div style="font-size:12px;color:#8b949e;margin-bottom:8px;text-transform:uppercase;letter-spacing:.5px;">{label}</div>
  <div style="font-size:26px;font-weight:700;color:{color};">${price:.2f}</div>
  <div style="font-size:14px;color:{color};font-weight:600;margin-top:2px;">{upside:+.1f}%</div>
  {note}
</div>""", unsafe_allow_html=True)

                # ── 3. EXPECTED RETURNS TABLE ─────────────────────────────────
                st.markdown("---")
                st.markdown("##### Expected Returns by Time Horizon")
                st.markdown("""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:14px 16px;margin-bottom:10px;font-size:13px;color:#c9d1d9;line-height:1.7;">
Forecasted returns are synthesized from <strong>four analytical reference lenses</strong>: technical momentum
(trend & RSI signals), fundamental valuation gap (intrinsic value vs market price), historical mean-reversion
patterns, and analyst consensus targets. Note: P/E and Forward P/E are correlated inputs and are not treated
as fully independent signals. <strong>Point Estimate</strong> is the probability-weighted expected return.
<strong>80% Range</strong> is the confidence interval. <strong>Confidence</strong> reflects convergence
across lenses — short-horizon estimates are more technically driven while long-horizon estimates weight
intrinsic fundamental value more heavily.
</div>""", unsafe_allow_html=True)

                _ret_rows = ""
                for _period, _fc in forecasts.items():
                    _ret  = _fc['point_estimate']
                    _rc   = "#3fb950" if _ret > 5 else "#d29922" if _ret > 0 else "#f85149"
                    _conf = _fc['confidence']
                    _cc   = "#3fb950" if _conf == "High" else "#d29922" if _conf == "Medium" else "#f85149"
                    _ret_rows += (
                        f"<tr style='border-bottom:1px solid #21262d;'>"
                        f"<td style='padding:10px 12px;font-size:13px;color:#e6edf3;font-weight:600;'>{_period}</td>"
                        f"<td style='padding:10px 12px;font-size:16px;color:{_rc};font-weight:700;text-align:center;'>{_ret:+.1f}%</td>"
                        f"<td style='padding:10px 12px;font-size:12px;color:#8b949e;text-align:center;'>{_fc['range_low']:+.1f}% \u2192 {_fc['range_high']:+.1f}%</td>"
                        f"<td style='padding:10px 12px;font-size:13px;color:#58a6ff;font-weight:600;text-align:center;'>${_fc['price_target']:.2f}</td>"
                        f"<td style='padding:10px 12px;text-align:center;'><span style='background:{_cc}22;border:1px solid {_cc};color:{_cc};border-radius:4px;padding:3px 10px;font-size:11px;font-weight:700;'>{_conf}</span></td>"
                        f"<td style='padding:10px 12px;font-size:12px;color:#8b949e;text-align:center;'>{_fc['probability']}</td>"
                        f"</tr>"
                    )
                st.markdown(
                    "<div style='background:#0d1117;border:1px solid #30363d;border-radius:8px;overflow:hidden;margin-bottom:8px;'>"
                    "<table style='width:100%;border-collapse:collapse;'>"
                    "<thead><tr style='background:#161b22;border-bottom:2px solid #30363d;'>"
                    "<th style='padding:10px 12px;font-size:11px;color:#8b949e;text-align:left;text-transform:uppercase;'>Period</th>"
                    "<th style='padding:10px 12px;font-size:11px;color:#8b949e;text-align:center;text-transform:uppercase;'>Expected Return</th>"
                    "<th style='padding:10px 12px;font-size:11px;color:#8b949e;text-align:center;text-transform:uppercase;'>80% Range</th>"
                    "<th style='padding:10px 12px;font-size:11px;color:#8b949e;text-align:center;text-transform:uppercase;'>Price Target</th>"
                    "<th style='padding:10px 12px;font-size:11px;color:#8b949e;text-align:center;text-transform:uppercase;'>Confidence</th>"
                    "<th style='padding:10px 12px;font-size:11px;color:#8b949e;text-align:center;text-transform:uppercase;'>Probability</th>"
                    f"</tr></thead><tbody>{_ret_rows}</tbody></table></div>",
                    unsafe_allow_html=True)

                # ── 4. INVESTMENT RATIONALE ───────────────────────────────────
                st.markdown("---")
                st.markdown("##### Investment Rationale")
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #30363d;border-radius:8px;padding:18px 20px;font-size:14px;color:#c9d1d9;line-height:1.9;">
{recommendation['rationale']}
</div>""", unsafe_allow_html=True)

                # ── 5. CATALYSTS & RISKS ──────────────────────────────────────
                st.markdown("---")
                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    st.markdown("##### \U0001f7e2 Bullish Catalysts")
                    if recommendation.get('bullish_drivers'):
                        _items = "".join(
                            f"<div style='padding:7px 0;border-bottom:1px solid #21262d;font-size:13px;color:#c9d1d9;'>"
                            f"<span style='color:#3fb950;margin-right:6px;'>\u2713</span>{d}</div>"
                            for d in recommendation['bullish_drivers']
                        )
                        st.markdown(f"<div style='background:#0d1117;border:1px solid #3fb95099;border-radius:8px;padding:10px 14px;'>{_items}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No significant bullish factors identified.")
                with b_col2:
                    st.markdown("##### \U0001f534 Key Risks")
                    if recommendation.get('bearish_risks'):
                        _items = "".join(
                            f"<div style='padding:7px 0;border-bottom:1px solid #21262d;font-size:13px;color:#c9d1d9;'>"
                            f"<span style='color:#f85149;margin-right:6px;'>\u26a0</span>{r}</div>"
                            for r in recommendation['bearish_risks']
                        )
                        st.markdown(f"<div style='background:#0d1117;border:1px solid #f8514999;border-radius:8px;padding:10px 14px;'>{_items}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No significant risk factors identified.")

                # ── 6. TRADE INVALIDATION ─────────────────────────────────────
                st.markdown("---")
                st.markdown(f"""
<div style="background:#0d1117;border:1px solid #d29922;border-radius:8px;padding:16px 18px;">
  <div style="font-size:11px;color:#d29922;font-weight:700;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;">\u26a0\ufe0f Trade Invalidation Criteria</div>
  <div style="font-size:13px;color:#c9d1d9;line-height:1.8;">{recommendation['invalidation']}</div>
</div>""", unsafe_allow_html=True)

                st.markdown("""
<div style="font-size:11px;color:#6e7681;text-align:center;padding:18px 0 4px 0;border-top:1px solid #21262d;margin-top:18px;">
  <strong>Disclaimer:</strong> Educational purposes only. Not financial or investment advice.
  Data sourced from Yahoo Finance with a 15\u201320 minute delay.
</div>""", unsafe_allow_html=True)

