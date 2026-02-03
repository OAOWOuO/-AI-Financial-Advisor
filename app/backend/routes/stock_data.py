"""Stock data endpoint for fetching current prices and performance tracking."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import os

router = APIRouter(prefix="/stock-data")


class StockPrice(BaseModel):
    ticker: str
    current_price: float
    previous_close: float
    change: float
    change_percent: float
    timestamp: str


class PerformanceCheck(BaseModel):
    ticker: str
    signal: str  # BULLISH, BEARISH, NEUTRAL
    signal_date: str
    signal_price: Optional[float]
    current_price: float
    price_change: float
    price_change_percent: float
    signal_correct: bool
    days_since_signal: int


@router.get("/prices/{tickers}")
async def get_stock_prices(tickers: str) -> Dict[str, StockPrice]:
    """Get current stock prices for given tickers (comma-separated)."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")]
    results = {}

    try:
        import yfinance as yf

        for ticker in ticker_list:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                hist = stock.history(period="2d")

                if len(hist) >= 1:
                    current = hist['Close'].iloc[-1]
                    previous = hist['Close'].iloc[-2] if len(hist) >= 2 else current
                    change = current - previous
                    change_pct = (change / previous) * 100 if previous > 0 else 0

                    results[ticker] = StockPrice(
                        ticker=ticker,
                        current_price=round(current, 2),
                        previous_close=round(previous, 2),
                        change=round(change, 2),
                        change_percent=round(change_pct, 2),
                        timestamp=datetime.now().isoformat(),
                    )
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                continue

    except ImportError:
        # yfinance not installed, return mock data
        for ticker in ticker_list:
            results[ticker] = StockPrice(
                ticker=ticker,
                current_price=150.00,
                previous_close=148.00,
                change=2.00,
                change_percent=1.35,
                timestamp=datetime.now().isoformat(),
            )

    return results


@router.post("/check-performance")
async def check_performance(data: dict) -> List[PerformanceCheck]:
    """Check if signals were correct based on price movement."""
    results = []

    try:
        import yfinance as yf

        for item in data.get("signals", []):
            ticker = item.get("ticker", "").upper()
            signal = item.get("signal", "NEUTRAL").upper()
            signal_date = item.get("signal_date", datetime.now().isoformat())
            signal_price = item.get("signal_price")

            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1mo")

                if len(hist) > 0:
                    current_price = hist['Close'].iloc[-1]

                    # If no signal price provided, estimate from history
                    if not signal_price:
                        signal_price = hist['Close'].iloc[0] if len(hist) > 5 else current_price

                    price_change = current_price - signal_price
                    price_change_pct = (price_change / signal_price) * 100 if signal_price > 0 else 0

                    # Determine if signal was correct
                    signal_correct = False
                    if signal == "BULLISH" and price_change > 0:
                        signal_correct = True
                    elif signal == "BEARISH" and price_change < 0:
                        signal_correct = True
                    elif signal == "NEUTRAL" and abs(price_change_pct) < 2:
                        signal_correct = True

                    # Calculate days since signal
                    try:
                        signal_dt = datetime.fromisoformat(signal_date.replace('Z', '+00:00'))
                        days_since = (datetime.now() - signal_dt.replace(tzinfo=None)).days
                    except:
                        days_since = 0

                    results.append(PerformanceCheck(
                        ticker=ticker,
                        signal=signal,
                        signal_date=signal_date,
                        signal_price=round(signal_price, 2),
                        current_price=round(current_price, 2),
                        price_change=round(price_change, 2),
                        price_change_percent=round(price_change_pct, 2),
                        signal_correct=signal_correct,
                        days_since_signal=days_since,
                    ))
            except Exception as e:
                print(f"Error checking performance for {ticker}: {e}")
                continue

    except ImportError:
        # Return mock data if yfinance not installed
        for item in data.get("signals", []):
            results.append(PerformanceCheck(
                ticker=item.get("ticker", "UNKNOWN"),
                signal=item.get("signal", "NEUTRAL"),
                signal_date=item.get("signal_date", datetime.now().isoformat()),
                signal_price=100.0,
                current_price=102.5,
                price_change=2.5,
                price_change_percent=2.5,
                signal_correct=True,
                days_since_signal=7,
            ))

    return results
