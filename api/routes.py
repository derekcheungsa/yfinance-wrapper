from flask import Blueprint, jsonify, request
from api.extensions import cache
import yfinance as yf
from .utils import validate_ticker, RateLimiter
import datetime
import pandas as pd
import numpy as np

api_bp = Blueprint('api', __name__)
rate_limiter = RateLimiter(requests=100, window=60)  # 100 requests per minute

def validate_numeric(value, fallback=None):
    """Validate and convert numeric values"""
    try:
        if value is not None:
            return float(value)
        return fallback
    except (ValueError, TypeError):
        return fallback

# Add the basic stock endpoint
@api_bp.route('/stock/<ticker>')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_stock_data(ticker):
    """Get current stock data"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price with fallback
        current_price = validate_numeric(info.get('currentPrice'))
        if current_price is None:
            current_price = validate_numeric(info.get('regularMarketPrice'))
        
        # Calculate price change and percentage
        previous_close = validate_numeric(info.get('previousClose'))
        price_change = None
        price_change_percent = None
        
        if current_price is not None and previous_close is not None:
            price_change = current_price - previous_close
            if previous_close != 0:
                price_change_percent = (price_change / previous_close) * 100

        stock_data = {
            'symbol': ticker,
            'price': {
                'current': current_price,
                'previous_close': previous_close,
                'change': validate_numeric(price_change),
                'change_percent': validate_numeric(price_change_percent),
                'day_high': validate_numeric(info.get('dayHigh')),
                'day_low': validate_numeric(info.get('dayLow')),
                'fifty_two_week_high': validate_numeric(info.get('fiftyTwoWeekHigh')),
                'fifty_two_week_low': validate_numeric(info.get('fiftyTwoWeekLow'))
            },
            'volume': {
                'current': validate_numeric(info.get('volume')),
                'average': validate_numeric(info.get('averageVolume')),
                'average_10_days': validate_numeric(info.get('averageVolume10days'))
            },
            'market_cap': validate_numeric(info.get('marketCap')),
            'data_quality': {
                'has_price_data': current_price is not None,
                'has_volume_data': info.get('volume') is not None,
                'has_market_cap': info.get('marketCap') is not None
            }
        }
        
        return jsonify(stock_data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch stock data: {str(e)}"), 500
