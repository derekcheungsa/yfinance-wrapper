from flask import Blueprint, jsonify, request
from api.extensions import cache
import yfinance as yf
from .utils import validate_ticker, RateLimiter
import datetime
import pandas as pd
import numpy as np

# Define Blueprint and rate limiter
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

def calculate_change_values(current_price, previous_close):
    """Calculate change and change percentage from current and previous prices"""
    if current_price is None or previous_close is None or previous_close == 0:
        return None, None
    
    change = current_price - previous_close
    change_percent = (change / previous_close) * 100
    return change, change_percent

def get_historical_changes(stock, ticker):
    """Get changes using historical data as fallback"""
    try:
        # Get today's data and yesterday's close
        history = stock.history(period="2d")
        if len(history) < 2:
            return None, None
        
        current_price = history['Close'].iloc[-1]
        previous_close = history['Close'].iloc[-2]
        
        return calculate_change_values(current_price, previous_close)
    except Exception:
        return None, None

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
        
        # Get primary values
        current_price = validate_numeric(info.get('currentPrice'))
        previous_close = validate_numeric(info.get('previousClose'))
        
        # Try primary change calculations
        change = validate_numeric(info.get('regularMarketChange'))
        change_percent = validate_numeric(info.get('regularMarketChangePercent'))
        
        # If change values are None, try calculating from current and previous prices
        if (change is None or change_percent is None) and current_price is not None and previous_close is not None:
            change, change_percent = calculate_change_values(current_price, previous_close)
        
        # If still None, try alternative fields
        if change is None:
            change = validate_numeric(info.get('regularMarketPrice')) - validate_numeric(info.get('regularMarketPreviousClose'))
        
        # If still None, try historical data
        if (change is None or change_percent is None) and current_price is not None:
            hist_change, hist_change_percent = get_historical_changes(stock, ticker)
            change = change if change is not None else hist_change
            change_percent = change_percent if change_percent is not None else hist_change_percent
        
        # Prepare response data
        data = {
            'symbol': ticker,
            'price': current_price,
            'change': change,
            'change_percent': change_percent,
            'volume': validate_numeric(info.get('regularMarketVolume')),
            'market_cap': validate_numeric(info.get('marketCap')),
            'previous_close': previous_close,
            'timestamp': datetime.datetime.now().isoformat(),
            'data_quality': {
                'price_source': 'real_time' if current_price is not None else 'unavailable',
                'change_calculation': 'real_time' if info.get('regularMarketChange') is not None 
                                    else 'calculated' if change is not None 
                                    else 'unavailable'
            }
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch stock data: {str(e)}"), 500

@api_bp.route('/stock/<ticker>/history')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_stock_history(ticker):
    """Get historical stock data"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    period = request.args.get('period', '6mo')
    interval = request.args.get('interval', '1d')
    
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period, interval=interval)
        
        data = {
            'symbol': ticker,
            'period': period,
            'interval': interval,
            'data': [{
                'date': index.isoformat(),
                'open': validate_numeric(row['Open']),
                'high': validate_numeric(row['High']),
                'low': validate_numeric(row['Low']),
                'close': validate_numeric(row['Close']),
                'volume': validate_numeric(row['Volume'])
            } for index, row in history.iterrows()]
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch historical data: {str(e)}"), 500

# [Rest of the endpoints remain the same as in the original code]
# Includes: get_moving_averages, get_rsi, get_statistics, get_company_info, get_market_summary, get_options_chain