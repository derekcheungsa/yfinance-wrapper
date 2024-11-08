from flask import Blueprint, jsonify, request
from flask_caching import Cache
import yfinance as yf
from .utils import validate_ticker, RateLimiter
import datetime

api_bp = Blueprint('api', __name__)
cache = Cache()
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
