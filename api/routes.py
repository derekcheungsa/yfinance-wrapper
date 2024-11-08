from flask import Blueprint, jsonify, request
from api.extensions import cache
import yfinance as yf
from .utils import validate_ticker, RateLimiter
import datetime
import pandas as pd
import numpy as np

api_bp = Blueprint('api', __name__)
rate_limiter = RateLimiter(requests=100, window=60)  # 100 requests per minute

@api_bp.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="Rate limit exceeded. Please try again later."), 429

@api_bp.errorhandler(400)
def bad_request_handler(e):
    return jsonify(error=str(e)), 400

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
        
        return jsonify({
            'symbol': ticker,
            'name': info.get('longName'),
            'price': info.get('currentPrice'),
            'change': info.get('regularMarketChangePercent'),
            'market_cap': info.get('marketCap'),
            'volume': info.get('volume')
        })
    except Exception as e:
        return jsonify(error=str(e)), 500

@api_bp.route('/stock/<ticker>/history')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_stock_history(ticker):
    """Get historical stock data"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    period = request.args.get('period', '1mo')
    interval = request.args.get('interval', '1d')
    
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period, interval=interval)
        
        return jsonify({
            'symbol': ticker,
            'history': [
                {
                    'date': index.strftime('%Y-%m-%d'),
                    'open': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close': row['Close'],
                    'volume': row['Volume']
                }
                for index, row in history.iterrows()
            ]
        })
    except Exception as e:
        return jsonify(error=str(e)), 500

@api_bp.route('/stock/<ticker>/analysis/moving-averages')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_moving_averages(ticker):
    """Get moving averages analysis"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    period = request.args.get('period', '6mo')
    short_window = int(request.args.get('short_window', '20'))
    long_window = int(request.args.get('long_window', '50'))
    
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        # Calculate SMAs and EMAs
        sma_short = history['Close'].rolling(window=short_window).mean()
        sma_long = history['Close'].rolling(window=long_window).mean()
        ema_short = history['Close'].ewm(span=short_window).mean()
        ema_long = history['Close'].ewm(span=long_window).mean()
        
        return jsonify({
            'symbol': ticker,
            'analysis': [
                {
                    'date': index.strftime('%Y-%m-%d'),
                    'close': row['Close'],
                    f'SMA_{short_window}': sma_short[idx],
                    f'SMA_{long_window}': sma_long[idx],
                    f'EMA_{short_window}': ema_short[idx],
                    f'EMA_{long_window}': ema_long[idx]
                }
                for idx, (index, row) in enumerate(history.iterrows())
            ]
        })
    except Exception as e:
        return jsonify(error=str(e)), 500

def calculate_rsi(data, periods=14):
    """Calculate RSI manually"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@api_bp.route('/stock/<ticker>/analysis/rsi')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_rsi_analysis(ticker):
    """Get RSI (Relative Strength Index) analysis"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    period = request.args.get('period', '6mo')
    window = int(request.args.get('window', '14'))
    
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        # Calculate RSI manually
        rsi = calculate_rsi(history['Close'], window)
        
        return jsonify({
            'symbol': ticker,
            'analysis': [
                {
                    'date': index.strftime('%Y-%m-%d'),
                    'close': row['Close'],
                    'rsi': float(rsi[idx]) if not pd.isna(rsi[idx]) else None
                }
                for idx, (index, row) in enumerate(history.iterrows())
            ]
        })
    except Exception as e:
        return jsonify(error=str(e)), 500

@api_bp.route('/stock/<ticker>/analysis/statistics')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_statistics_analysis(ticker):
    """Get basic statistical analysis"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    period = request.args.get('period', '1y')
    
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        # Calculate daily returns
        daily_returns = history['Close'].pct_change()
        
        # Calculate various statistics
        stats = {
            'symbol': ticker,
            'period': period,
            'statistics': {
                'mean_return': float(daily_returns.mean()),
                'std_dev': float(daily_returns.std()),
                'annualized_volatility': float(daily_returns.std() * np.sqrt(252)),
                'min_price': float(history['Close'].min()),
                'max_price': float(history['Close'].max()),
                'current_price': float(history['Close'].iloc[-1]),
                'price_change': float((history['Close'].iloc[-1] / history['Close'].iloc[0] - 1) * 100),
                'trading_days': len(history)
            }
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify(error=str(e)), 500

@api_bp.route('/stock/<ticker>/info')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_company_info(ticker):
    """Get company information"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return jsonify({
            'symbol': ticker,
            'name': info.get('longName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'website': info.get('website'),
            'description': info.get('longBusinessSummary'),
            'employees': info.get('fullTimeEmployees')
        })
    except Exception as e:
        return jsonify(error=str(e)), 500

@api_bp.route('/market/summary')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_market_summary():
    """Get market summary"""
    indices = ['^GSPC', '^DJI', '^IXIC']  # S&P 500, Dow Jones, NASDAQ
    
    try:
        summary = {}
        for index in indices:
            ticker = yf.Ticker(index)
            info = ticker.info
            summary[index] = {
                'name': info.get('shortName'),
                'price': info.get('regularMarketPrice'),
                'change': info.get('regularMarketChangePercent')
            }
        
        return jsonify(summary)
    except Exception as e:
        return jsonify(error=str(e)), 500