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
            rsi_score = 5
            signals.append({
                "category": "Momentum", "indicator": "RSI",
                "signal": "NEUTRAL", "score": "+5",
                "detail": f"RSI at {rsi:.1f} - Neutral zone",
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
                "detail": "MACD above signal with expanding histogram",
                "threshold": "MACD > Signal, Histogram increasing"
            })
        elif latest['MACD'] < latest['MACD_Signal'] and latest['MACD_Hist'] < prev['MACD_Hist']:
            macd_score = -10
            signals.append({
                "category": "Momentum", "indicator": "MACD",
                "signal": "BEARISH", "score": "-10",
                "detail": "MACD below signal with expanding histogram",
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
            vol_score = 0
            signals.append({
                "category": "Volume", "indicator": "Volume",
                "signal": "CAUTION", "score": "0",
                "detail": f"Low volume ({vol_ratio:.1f}x avg) - Weak conviction",
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
            signals.append({
                "category": "Volume", "indicator": "OBV",
                "signal": "NEUTRAL", "score": "0",
                "detail": f"OBV stable around average",
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

    # PEG Ratio (10 points)
    peg = data.get('peg_ratio')
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
            signals.append({
                "category": "Profitability", "metric": "Return on Equity",
                "value": f"{roe_pct:.1f}%", "signal": "EXCELLENT", "score": "+10",
                "detail": "Superior capital efficiency",
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
    de_ratio = data.get('debt_to_equity')
    if de_ratio is not None:
        if de_ratio < 0.3:
            health_score += 10
            signals.append({
                "category": "Financial Health", "metric": "Debt/Equity",
                "value": f"{de_ratio:.2f}", "signal": "STRONG", "score": "+10",
                "detail": "Very low leverage - conservative balance sheet",
                "benchmark": "Conservative: <0.3"
            })
        elif de_ratio < 0.7:
            health_score += 7
            signals.append({
                "category": "Financial Health", "metric": "Debt/Equity",
                "value": f"{de_ratio:.2f}", "signal": "HEALTHY", "score": "+7",
                "detail": "Moderate leverage",
                "benchmark": "Healthy: 0.3-0.7"
            })
        elif de_ratio < 1.5:
            health_score += 3
            signals.append({
                "category": "Financial Health", "metric": "Debt/Equity",
                "value": f"{de_ratio:.2f}", "signal": "ELEVATED", "score": "+3",
                "detail": "Higher leverage - monitor interest coverage",
                "benchmark": "Elevated: 0.7-1.5"
            })
        else:
            health_score -= 5
            signals.append({
                "category": "Financial Health", "metric": "Debt/Equity",
                "value": f"{de_ratio:.2f}", "signal": "HIGH RISK", "score": "-5",
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

    # Determine fundamental rating
    if score_pct >= 45:
        fund_rating = ("STRONG BUY", "#00ff00")
    elif score_pct >= 25:
        fund_rating = ("BUY", "#3fb950")
    elif score_pct >= 5:
        fund_rating = ("HOLD", "#d29922")
    elif score_pct >= -15:
        fund_rating = ("SELL", "#f85149")
    else:
        fund_rating = ("STRONG SELL", "#ff0000")

    return {
        "signals": signals,
        "total_score": total_score,
        "max_score": max_score,
        "score_pct": score_pct,
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

        dcf_bear = simple_dcf(fcf_per_share, 0.05, 0.12)
        dcf_base = simple_dcf(fcf_per_share, 0.08, 0.10)
        dcf_bull = simple_dcf(fcf_per_share, 0.12, 0.08)

        results['dcf_valuation'] = {
            'bear': dcf_bear,
            'base': dcf_base,
            'bull': dcf_bull,
            'fcf_per_share': fcf_per_share
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

    # Adjust based on technical score
    tech_adjustment = tech_analysis['score_pct'] / 100 * 5  # +/- 5% based on technicals

    # Adjust based on fundamental score
    fund_adjustment = fund_analysis['score_pct'] / 100 * 5  # +/- 5% based on fundamentals

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

        # Confidence based on signal strength
        combined_score = (tech_analysis['score_pct'] + fund_analysis['score_pct']) / 2

        if abs(combined_score) > 40:
            confidence = "HIGH"
            probability = "65-75%"
        elif abs(combined_score) > 20:
            confidence = "MEDIUM"
            probability = "55-65%"
        else:
            confidence = "LOW"
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
def generate_recommendation(data: Dict, tech_analysis: Dict, fund_analysis: Dict, valuation: Dict, forecasts: Dict) -> Dict:
    """Generate final institutional-grade recommendation."""
    price = data['price']

    # Combined score (60% fundamental, 40% technical)
    combined_score = fund_analysis['score_pct'] * 0.6 + tech_analysis['score_pct'] * 0.4

    # Determine action
    if combined_score >= 35:
        action = "STRONG BUY"
        action_color = "#00ff00"
    elif combined_score >= 15:
        action = "BUY"
        action_color = "#3fb950"
    elif combined_score >= -5:
        action = "HOLD"
        action_color = "#d29922"
    elif combined_score >= -25:
        action = "SELL"
        action_color = "#f85149"
    else:
        action = "STRONG SELL"
        action_color = "#ff0000"

    # Trade decision
    if action in ["STRONG BUY", "BUY"]:
        trade_decision = "INITIATE LONG POSITION"
    elif action in ["STRONG SELL", "SELL"]:
        trade_decision = "AVOID / EXIT POSITION"
    else:
        trade_decision = "NO ACTION - WAIT FOR BETTER ENTRY"

    # Price target and upside
    if 'composite' in valuation:
        target_price = valuation['composite']['target_mid']
        upside = valuation['composite']['upside_mid']
        target_low = valuation['composite']['target_low']
        target_high = valuation['composite']['target_high']
    else:
        target_price = price
        upside = 0
        target_low = price * 0.9
        target_high = price * 1.1

    # Key drivers (bullish factors)
    bullish_drivers = []
    bearish_risks = []

    for signal in tech_analysis['signals'] + fund_analysis['signals']:
        if signal.get('signal') in ['BULLISH', 'STRONG', 'UNDERVALUED', 'HIGH GROWTH', 'EXCELLENT']:
            bullish_drivers.append(f"{signal.get('metric', signal.get('indicator', 'N/A'))}: {signal.get('detail', '')}")
        elif signal.get('signal') in ['BEARISH', 'WEAK', 'OVERVALUED', 'DECLINING', 'HIGH RISK', 'LOSS']:
            bearish_risks.append(f"{signal.get('metric', signal.get('indicator', 'N/A'))}: {signal.get('detail', '')}")

    # Rationale
    rationale = f"""
**Investment Thesis:**
{data['name']} ({data['ticker']}) rates as a **{action}** based on combined technical and fundamental analysis.

**Key Drivers:**
{''.join(['• ' + d + chr(10) for d in bullish_drivers[:3]]) if bullish_drivers else '• No significant bullish factors identified'}

**Key Risks:**
{''.join(['• ' + r + chr(10) for r in bearish_risks[:3]]) if bearish_risks else '• No significant bearish factors identified'}

**Valuation:**
Current price of ${price:.2f} {'represents an attractive entry point' if upside > 15 else 'is fairly valued' if abs(upside) < 10 else 'appears overvalued'} with a target of ${target_price:.2f} ({upside:+.1f}% potential).
"""

    # Invalidation criteria
    if action in ["STRONG BUY", "BUY"]:
        invalidation = f"Exit if price falls below ${price * 0.9:.2f} (-10%) or if key technical support breaks."
    elif action in ["STRONG SELL", "SELL"]:
        invalidation = f"Reconsider if price breaks above ${price * 1.1:.2f} (+10%) with strong volume."
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
        if st.button("← Back", key="back_analyzer"):
            st.session_state.current_view = 'home'
            st.rerun()
    with col_title:
        st.write("# 📈 Institutional Stock Analyzer")
        st.caption("CFA-Style Technical & Fundamental Analysis with Price Forecasting")

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
        recommendation = generate_recommendation(data, tech_analysis, fund_analysis, valuation, forecasts)
        supports, resistances = identify_support_resistance(hist)

    # ── TWO-COLUMN LAYOUT: LEFT (input + company info) | RIGHT (analysis tabs) ─
    col_left, col_right = st.columns([1, 2])

    with col_left:
        ticker = st.text_input("Ticker Symbol", value="AAPL", key="inst_ticker",
                               placeholder="e.g. AAPL, MSFT, NVDA")
        ticker = ticker.strip().upper()
        analyze_btn = st.button("🔍 Run Full Analysis", type="primary", use_container_width=True)

        if analyze_btn and ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                st.session_state.inst_data = fetch_comprehensive_data(ticker)
                st.rerun()

        if not has_data:
            st.info("Enter a ticker and click **Run Full Analysis**.")
        else:
            st.divider()
            st.write(f"### {data['name']}")
            st.caption(f"{data['sector']} · {data['industry']}")
            change_pct = ((data['price'] - data.get('prev_close', data['price'])) / data.get('prev_close', data['price']) * 100) if data.get('prev_close') else 0
            st.metric("Price", f"${data['price']:.2f}", f"{change_pct:+.2f}%")
            st.write(f"**Market Cap:** ${data['market_cap']/1e9:.1f}B")
            st.write(f"**52W Range:** ${data['low_52w']:.0f} – ${data['high_52w']:.0f}")
            if data['high_52w'] > data['low_52w']:
                pos = (data['price'] - data['low_52w']) / (data['high_52w'] - data['low_52w'])
                st.progress(min(1.0, max(0.0, pos)), text=f"{pos*100:.0f}% of range")
            st.divider()
            st.markdown(f"""
<div style="background:{recommendation['action_color']}22;border:2px solid {recommendation['action_color']};border-radius:8px;padding:12px;text-align:center;">
  <div style="font-size:22px;font-weight:700;color:{recommendation['action_color']};">{recommendation['action']}</div>
  <div style="font-size:13px;color:#c9d1d9;">Target: ${recommendation['target_price']:.2f} ({recommendation['upside']:+.1f}%)</div>
  <div style="font-size:11px;color:#8b949e;">Score: {recommendation['combined_score']:.0f}</div>
</div>
""", unsafe_allow_html=True)

    with col_right:
        tab_tech, tab_fund, tab_conclusion = st.tabs(
            ["📊 Technical Analysis", "📋 Fundamental Analysis", "🎯 Conclusion & Forecast"]
        )

        # ── TECHNICAL TAB ──────────────────────────────────────────────────────
        with tab_tech:
            if not has_data:
                st.info("Enter a ticker and click **Run Full Analysis** to generate the report.")
            else:
                st.write("### Technical Analysis Summary")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Overall", f"{tech_analysis['score_pct']:.0f}%", tech_analysis['rating'])
                with col2:
                    st.metric("Trend", f"{tech_analysis['breakdown']['trend']['score']}/{tech_analysis['breakdown']['trend']['max']}")
                with col3:
                    st.metric("Momentum", f"{tech_analysis['breakdown']['momentum']['score']}/{tech_analysis['breakdown']['momentum']['max']}")
                with col4:
                    st.metric("Volatility", f"{tech_analysis['breakdown']['volatility']['score']}/{tech_analysis['breakdown']['volatility']['max']}")
                with col5:
                    st.metric("Volume", f"{tech_analysis['breakdown']['volume']['score']}/{tech_analysis['breakdown']['volume']['max']}")

                st.divider()
                st.write("### Price Chart with Moving Averages")
                chart_data = tech_df.reset_index()
                chart_data['Date'] = pd.to_datetime(chart_data['Date']).dt.tz_localize(None)

                price_chart = alt.Chart(chart_data).mark_line(color='#58a6ff', strokeWidth=2).encode(
                    x=alt.X('Date:T', title=None),
                    y=alt.Y('Close:Q', title='Price ($)', scale=alt.Scale(zero=False)),
                    tooltip=[alt.Tooltip('Date:T', format='%Y-%m-%d'), alt.Tooltip('Close:Q', format='$.2f', title='Price')]
                )
                ma50  = alt.Chart(chart_data).mark_line(color='#f0883e', strokeWidth=1.5, strokeDash=[5,5]).encode(x='Date:T', y='SMA_50:Q')
                ma200 = alt.Chart(chart_data).mark_line(color='#a371f7', strokeWidth=1.5, strokeDash=[5,5]).encode(x='Date:T', y='SMA_200:Q')
                if supports:
                    support_df = pd.DataFrame({'y': [supports[-1]]})
                    price_chart = price_chart + alt.Chart(support_df).mark_rule(color='#3fb950', strokeDash=[3,3]).encode(y='y:Q')
                if resistances:
                    resist_df = pd.DataFrame({'y': [resistances[-1]]})
                    price_chart = price_chart + alt.Chart(resist_df).mark_rule(color='#f85149', strokeDash=[3,3]).encode(y='y:Q')
                st.altair_chart((price_chart + ma50 + ma200).properties(height=300).configure_view(strokeWidth=0).configure(background='#161b22'), use_container_width=True)
                st.caption("🔵 Price  🟠 SMA50  🟣 SMA200  🟢 Support  🔴 Resistance")

                col_rsi, col_macd = st.columns(2)
                with col_rsi:
                    st.write("### RSI (14)")
                    rsi_chart = alt.Chart(chart_data.tail(100)).mark_line(color='#58a6ff').encode(
                        x=alt.X('Date:T', title=None),
                        y=alt.Y('RSI:Q', scale=alt.Scale(domain=[0, 100]), title='RSI')
                    )
                    rules = alt.Chart(pd.DataFrame({'y': [30, 70]})).mark_rule(strokeDash=[3,3], color='#6e7681').encode(y='y:Q')
                    st.altair_chart((rsi_chart + rules).properties(height=180).configure_view(strokeWidth=0).configure(background='#161b22'), use_container_width=True)
                    rsi_val = tech_df['RSI'].iloc[-1]
                    if rsi_val > 70:
                        st.error(f"RSI: {rsi_val:.1f} — OVERBOUGHT")
                    elif rsi_val < 30:
                        st.success(f"RSI: {rsi_val:.1f} — OVERSOLD")
                    else:
                        st.info(f"RSI: {rsi_val:.1f} — Neutral")

                with col_macd:
                    st.write("### MACD")
                    macd_c = alt.Chart(chart_data.tail(100)).mark_line(color='#58a6ff').encode(x='Date:T', y='MACD:Q')
                    sig_c  = alt.Chart(chart_data.tail(100)).mark_line(color='#f0883e').encode(x='Date:T', y='MACD_Signal:Q')
                    hist_c = alt.Chart(chart_data.tail(100)).mark_bar(opacity=0.5).encode(
                        x='Date:T', y='MACD_Hist:Q',
                        color=alt.condition(alt.datum.MACD_Hist > 0, alt.value('#3fb950'), alt.value('#f85149'))
                    )
                    st.altair_chart((hist_c + macd_c + sig_c).properties(height=180).configure_view(strokeWidth=0).configure(background='#161b22'), use_container_width=True)
                    if tech_df['MACD'].iloc[-1] > tech_df['MACD_Signal'].iloc[-1]:
                        st.success("MACD: Bullish (above signal)")
                    else:
                        st.warning("MACD: Bearish (below signal)")

                st.write("### Technical Signal Details")
                tech_signals_df = pd.DataFrame(tech_analysis['signals'])
                if not tech_signals_df.empty:
                    st.dataframe(
                        tech_signals_df[['category', 'indicator', 'signal', 'score', 'detail', 'threshold']],
                        hide_index=True, use_container_width=True
                    )
                st.caption("Source: Yahoo Finance (real-time market feed, 15-20 min delayed).")

        # ── FUNDAMENTAL TAB ────────────────────────────────────────────────────
        with tab_fund:
            if not has_data:
                st.info("Run an analysis to see fundamental data.")
            else:
                st.write("### Fundamental Analysis Summary")
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("Overall", f"{fund_analysis['score_pct']:.0f}%", fund_analysis['rating'])
                with col2:
                    st.metric("Valuation", f"{fund_analysis['breakdown']['valuation']['score']}/{fund_analysis['breakdown']['valuation']['max']}")
                with col3:
                    st.metric("Profitability", f"{fund_analysis['breakdown']['profitability']['score']}/{fund_analysis['breakdown']['profitability']['max']}")
                with col4:
                    st.metric("Growth", f"{fund_analysis['breakdown']['growth']['score']}/{fund_analysis['breakdown']['growth']['max']}")
                with col5:
                    st.metric("Health", f"{fund_analysis['breakdown']['health']['score']}/{fund_analysis['breakdown']['health']['max']}")

                st.divider()
                st.write("### Key Financial Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.write("**Valuation**")
                    st.write(f"P/E: {data.get('pe_ratio', 'N/A'):.1f}x" if data.get('pe_ratio') else "P/E: N/A")
                    st.write(f"Fwd P/E: {data.get('forward_pe', 'N/A'):.1f}x" if data.get('forward_pe') else "Fwd P/E: N/A")
                    st.write(f"PEG: {data.get('peg_ratio', 'N/A'):.2f}" if data.get('peg_ratio') else "PEG: N/A")
                    st.write(f"EV/EBITDA: {data.get('ev_ebitda', 'N/A'):.1f}x" if data.get('ev_ebitda') else "EV/EBITDA: N/A")
                with col2:
                    st.write("**Profitability**")
                    st.write(f"Op Margin: {data.get('operating_margin', 0)*100:.1f}%" if data.get('operating_margin') else "Op Margin: N/A")
                    st.write(f"Net Margin: {data.get('profit_margin', 0)*100:.1f}%" if data.get('profit_margin') else "Net Margin: N/A")
                    st.write(f"ROE: {data.get('roe', 0)*100:.1f}%" if data.get('roe') else "ROE: N/A")
                    st.write(f"ROA: {data.get('roa', 0)*100:.1f}%" if data.get('roa') else "ROA: N/A")
                with col3:
                    st.write("**Growth**")
                    st.write(f"Rev Growth: {data.get('revenue_growth', 0)*100:.1f}%" if data.get('revenue_growth') else "Rev Growth: N/A")
                    st.write(f"EPS Growth: {data.get('earnings_growth', 0)*100:.1f}%" if data.get('earnings_growth') else "EPS Growth: N/A")
                    st.write(f"EPS: ${data.get('eps', 0):.2f}" if data.get('eps') else "EPS: N/A")
                    st.write(f"Fwd EPS: ${data.get('forward_eps', 0):.2f}" if data.get('forward_eps') else "Fwd EPS: N/A")
                with col4:
                    st.write("**Financial Health**")
                    st.write(f"D/E: {data.get('debt_to_equity', 0):.2f}" if data.get('debt_to_equity') is not None else "D/E: N/A")
                    st.write(f"Current: {data.get('current_ratio', 0):.2f}" if data.get('current_ratio') else "Current: N/A")
                    fcf = data.get('free_cash_flow', 0)
                    st.write(f"FCF: ${fcf/1e9:.1f}B" if fcf else "FCF: N/A")
                    st.write(f"Cash: ${data.get('total_cash', 0)/1e9:.1f}B" if data.get('total_cash') else "Cash: N/A")

                st.divider()
                st.write("### Valuation Analysis")
                if valuation:
                    val_cols = st.columns(4)
                    if 'pe_valuation' in valuation:
                        with val_cols[0]:
                            st.write("**P/E Based**")
                            st.write(f"Bear: ${valuation['pe_valuation']['low']:.2f}")
                            st.write(f"Base: ${valuation['pe_valuation']['mid']:.2f}")
                            st.write(f"Bull: ${valuation['pe_valuation']['high']:.2f}")
                    if 'forward_pe_valuation' in valuation:
                        with val_cols[1]:
                            st.write("**Forward P/E**")
                            st.write(f"Bear: ${valuation['forward_pe_valuation']['low']:.2f}")
                            st.write(f"Base: ${valuation['forward_pe_valuation']['mid']:.2f}")
                            st.write(f"Bull: ${valuation['forward_pe_valuation']['high']:.2f}")
                    if 'analyst_target' in valuation:
                        with val_cols[2]:
                            st.write("**Analyst Consensus**")
                            st.write(f"Low: ${valuation['analyst_target']['low']:.2f}")
                            st.write(f"Mean: ${valuation['analyst_target']['mid']:.2f}")
                            st.write(f"High: ${valuation['analyst_target']['high']:.2f}")
                    if 'dcf_valuation' in valuation:
                        with val_cols[3]:
                            st.write("**DCF Model**")
                            st.write(f"Bear: ${valuation['dcf_valuation']['bear']:.2f}")
                            st.write(f"Base: ${valuation['dcf_valuation']['base']:.2f}")
                            st.write(f"Bull: ${valuation['dcf_valuation']['bull']:.2f}")
                    if 'composite' in valuation:
                        st.divider()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Target (Bear)", f"${valuation['composite']['target_low']:.2f}", f"{valuation['composite']['upside_low']:+.1f}%")
                        with col2:
                            st.metric("Target (Base)", f"${valuation['composite']['target_mid']:.2f}", f"{valuation['composite']['upside_mid']:+.1f}%")
                        with col3:
                            st.metric("Target (Bull)", f"${valuation['composite']['target_high']:.2f}", f"{valuation['composite']['upside_high']:+.1f}%")

                st.divider()
                st.write("### Fundamental Signal Details")
                fund_signals_df = pd.DataFrame(fund_analysis['signals'])
                if not fund_signals_df.empty:
                    st.dataframe(
                        fund_signals_df[['category', 'metric', 'value', 'signal', 'score', 'detail', 'benchmark']],
                        hide_index=True, use_container_width=True
                    )
                st.caption("Source: Yahoo Finance — SEC filings (10-K, 10-Q). Analyst estimates aggregated from major brokerages.")

        # ── CONCLUSION TAB ─────────────────────────────────────────────────────
        with tab_conclusion:
            if not has_data:
                st.info("Run an analysis to see the conclusion and forecast.")
            else:
                st.write("### Investment Recommendation")
                st.markdown(f"""
<div style="background:{recommendation['action_color']}22;border:2px solid {recommendation['action_color']};border-radius:12px;padding:25px;margin:10px 0;">
  <div style="display:flex;justify-content:space-between;align-items:center;">
    <div>
      <div style="font-size:36px;font-weight:700;color:{recommendation['action_color']};">{recommendation['action']}</div>
      <div style="font-size:16px;color:#c9d1d9;margin-top:5px;">{recommendation['trade_decision']}</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:13px;color:#8b949e;">12-Month Target</div>
      <div style="font-size:28px;font-weight:600;color:#e6edf3;">${recommendation['target_price']:.2f}</div>
      <div style="font-size:15px;color:{recommendation['action_color']};">{recommendation['upside']:+.1f}% Potential</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

                st.write("### Price Target Range")
                col1, col2, col3 = st.columns(3)
                with col1:
                    upside_low = (recommendation['target_low'] - data['price']) / data['price'] * 100
                    st.metric("Downside Case", f"${recommendation['target_low']:.2f}", f"{upside_low:+.1f}%")
                with col2:
                    st.metric("Base Case", f"${recommendation['target_price']:.2f}", f"{recommendation['upside']:+.1f}%")
                with col3:
                    upside_high = (recommendation['target_high'] - data['price']) / data['price'] * 100
                    st.metric("Upside Case", f"${recommendation['target_high']:.2f}", f"{upside_high:+.1f}%")

                st.divider()
                st.write("### Expected Returns by Time Horizon")
                returns_data = []
                for period, forecast in forecasts.items():
                    returns_data.append({
                        "Period": period,
                        "Expected Return": f"{forecast['point_estimate']:+.1f}%",
                        "Range": f"{forecast['range_low']:+.1f}% to {forecast['range_high']:+.1f}%",
                        "Price Target": f"${forecast['price_target']:.2f}",
                        "Confidence": forecast['confidence'],
                        "Probability": forecast['probability']
                    })
                st.dataframe(pd.DataFrame(returns_data), hide_index=True, use_container_width=True)

                st.divider()
                st.write("### Investment Rationale")
                st.markdown(recommendation['rationale'])

                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### 🟢 Bullish Factors")
                    if recommendation['bullish_drivers']:
                        for driver in recommendation['bullish_drivers']:
                            st.success(f"• {driver}")
                    else:
                        st.info("No significant bullish factors identified")
                with col2:
                    st.write("### 🔴 Key Risks")
                    if recommendation['bearish_risks']:
                        for risk in recommendation['bearish_risks']:
                            st.error(f"• {risk}")
                    else:
                        st.info("No significant risk factors identified")

                st.divider()
                st.write("### ⚠️ Trade Invalidation")
                st.warning(recommendation['invalidation'])
                st.markdown("""
---
<div style="font-size:11px;color:#6e7681;text-align:center;padding:10px 0;">
<strong>Disclaimer:</strong> Educational purposes only. Not financial advice.
Data from Yahoo Finance with 15-20 min delay.
</div>
""", unsafe_allow_html=True)

    # ── COMPACT AI CHAT — below both columns ─────────────────────────────────
    st.divider()
    st.write("### 💬 AI Financial Analyst Chat")

    _chat_key = f"chat_history_{data['ticker']}" if has_data else "chat_history_default"
    if _chat_key not in st.session_state:
        st.session_state[_chat_key] = []

    if has_data:
        st.caption(f"Chatting about **{data['ticker']} — {data['name']}**. Ask anything about the analysis.")
    else:
        st.caption("Run an analysis above to start chatting.")

    # Compact scrollable chat history
    _chat_box = st.container(height=320)
    with _chat_box:
        if not st.session_state[_chat_key]:
            st.markdown(
                "<p style='color:#6e7681;font-size:13px;text-align:center;padding-top:30px;'>"
                "No messages yet — ask a question below.</p>",
                unsafe_allow_html=True
            )
        for _msg in st.session_state[_chat_key]:
            with st.chat_message(_msg["role"]):
                st.markdown(_msg["content"])

    # Input bar always at bottom
    if _prompt := st.chat_input("Ask about this stock..."):
        st.session_state[_chat_key].append({"role": "user", "content": _prompt})
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
                                _placeholder.markdown(_response_text + "▌")
                        _placeholder.markdown(_response_text)

                st.session_state[_chat_key].append({"role": "assistant", "content": _response_text})

            except Exception as _e:
                _reply = f"Chat error: {_e}"
                st.session_state[_chat_key].append({"role": "assistant", "content": _reply})
                with _chat_box:
                    with st.chat_message("assistant"):
                        st.markdown(_reply)

    if st.session_state.get(_chat_key):
        if st.button("Clear Chat History", key="clear_chat_top"):
            st.session_state[_chat_key] = []
            st.rerun()

