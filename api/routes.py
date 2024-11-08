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
        
        data = {
            'symbol': ticker,
            'price': validate_numeric(info.get('currentPrice')),
            'change': validate_numeric(info.get('regularMarketChange')),
            'change_percent': validate_numeric(info.get('regularMarketChangePercent')),
            'volume': validate_numeric(info.get('regularMarketVolume')),
            'market_cap': validate_numeric(info.get('marketCap')),
            'timestamp': datetime.datetime.now().isoformat()
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

@api_bp.route('/stock/<ticker>/analysis/moving-averages')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_moving_averages(ticker):
    """Get moving averages analysis"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    period = request.args.get('period', '6mo')
    short_window = int(request.args.get('short_window', 20))
    long_window = int(request.args.get('long_window', 50))
    
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        # Calculate SMAs
        sma_short = history['Close'].rolling(window=short_window).mean()
        sma_long = history['Close'].rolling(window=long_window).mean()
        
        # Calculate EMAs
        ema_short = history['Close'].ewm(span=short_window).mean()
        ema_long = history['Close'].ewm(span=long_window).mean()
        
        data = {
            'symbol': ticker,
            'period': period,
            'indicators': {
                'sma': {
                    f'SMA_{short_window}': validate_numeric(sma_short.iloc[-1]),
                    f'SMA_{long_window}': validate_numeric(sma_long.iloc[-1])
                },
                'ema': {
                    f'EMA_{short_window}': validate_numeric(ema_short.iloc[-1]),
                    f'EMA_{long_window}': validate_numeric(ema_long.iloc[-1])
                }
            },
            'current_price': validate_numeric(history['Close'].iloc[-1])
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify(error=f"Failed to calculate moving averages: {str(e)}"), 500

@api_bp.route('/stock/<ticker>/analysis/rsi')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_rsi(ticker):
    """Get RSI analysis"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    period = request.args.get('period', '6mo')
    window = int(request.args.get('window', 14))
    
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        # Calculate daily price changes
        delta = history['Close'].diff()
        
        # Separate gains and losses
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        data = {
            'symbol': ticker,
            'period': period,
            'window': window,
            'rsi': validate_numeric(rsi.iloc[-1]),
            'rsi_values': {
                'last_5_days': [validate_numeric(x) for x in rsi.tail(5).tolist()]
            }
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify(error=f"Failed to calculate RSI: {str(e)}"), 500

@api_bp.route('/stock/<ticker>/analysis/statistics')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_statistics(ticker):
    """Get statistical analysis"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    period = request.args.get('period', '1y')
    
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        
        # Calculate returns
        returns = history['Close'].pct_change()
        
        data = {
            'symbol': ticker,
            'period': period,
            'statistics': {
                'volatility': validate_numeric(returns.std() * np.sqrt(252)),  # Annualized volatility
                'avg_daily_return': validate_numeric(returns.mean()),
                'max_drawdown': validate_numeric(((history['Close'].cummax() - history['Close']) / history['Close'].cummax()).max()),
                'sharpe_ratio': validate_numeric(returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else None,
                'skewness': validate_numeric(returns.skew()),
                'kurtosis': validate_numeric(returns.kurtosis())
            }
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify(error=f"Failed to calculate statistics: {str(e)}"), 500

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
        
        data = {
            'symbol': ticker,
            'name': info.get('longName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'description': info.get('longBusinessSummary'),
            'website': info.get('website'),
            'market_cap': validate_numeric(info.get('marketCap')),
            'employees': info.get('fullTimeEmployees'),
            'country': info.get('country'),
            'exchange': info.get('exchange')
        }
        
        return jsonify(data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch company info: {str(e)}"), 500

@api_bp.route('/market/summary')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_market_summary():
    """Get market summary"""
    try:
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow Jones, NASDAQ, Russell 2000
        
        data = {
            'indices': [],
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        for index in indices:
            try:
                ticker = yf.Ticker(index)
                info = ticker.info
                
                index_data = {
                    'symbol': index,
                    'name': info.get('shortName'),
                    'price': validate_numeric(info.get('regularMarketPrice')),
                    'change': validate_numeric(info.get('regularMarketChange')),
                    'change_percent': validate_numeric(info.get('regularMarketChangePercent')),
                    'volume': validate_numeric(info.get('regularMarketVolume'))
                }
                
                data['indices'].append(index_data)
            except Exception:
                continue
        
        return jsonify(data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch market summary: {str(e)}"), 500

# Keep the existing options chain endpoint
@api_bp.route('/stock/<ticker>/options')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_options_chain(ticker):
    """Get options chain data for a stock"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    try:
        stock = yf.Ticker(ticker)
        
        # Get current stock price for reference
        info = stock.info
        current_price = validate_numeric(info.get('currentPrice'))
        if current_price is None:
            current_price = validate_numeric(info.get('regularMarketPrice'))
        
        # Get available expiration dates
        expiration_dates = stock.options
        
        if not expiration_dates:
            return jsonify({
                'symbol': ticker,
                'error': 'No options data available',
                'data_quality': {'has_options_data': False}
            }), 404
        
        options_data = {
            'symbol': ticker,
            'current_price': current_price,
            'expiration_dates': [],
            'data_quality': {
                'has_options_data': True,
                'has_current_price': current_price is not None,
                'number_of_expiration_dates': len(expiration_dates)
            }
        }
        
        # Get options data for each expiration date
        for date in expiration_dates[:3]:  # Limit to first 3 dates to avoid timeout
            calls = stock.option_chain(date).calls
            puts = stock.option_chain(date).puts
            
            expiry_data = {
                'expiration_date': date,
                'days_to_expiration': (pd.Timestamp(date) - pd.Timestamp.now()).days,
                'calls': [],
                'puts': []
            }
            
            # Process calls
            for _, option in calls.iterrows():
                call_data = {
                    'strike': validate_numeric(option.get('strike')),
                    'last_price': validate_numeric(option.get('lastPrice')),
                    'bid': validate_numeric(option.get('bid')),
                    'ask': validate_numeric(option.get('ask')),
                    'volume': validate_numeric(option.get('volume')),
                    'open_interest': validate_numeric(option.get('openInterest')),
                    'implied_volatility': validate_numeric(option.get('impliedVolatility')),
                    'in_the_money': bool(option.get('inTheMoney')),
                    'contract_symbol': option.get('contractSymbol')
                }
                expiry_data['calls'].append(call_data)
            
            # Process puts
            for _, option in puts.iterrows():
                put_data = {
                    'strike': validate_numeric(option.get('strike')),
                    'last_price': validate_numeric(option.get('lastPrice')),
                    'bid': validate_numeric(option.get('bid')),
                    'ask': validate_numeric(option.get('ask')),
                    'volume': validate_numeric(option.get('volume')),
                    'open_interest': validate_numeric(option.get('openInterest')),
                    'implied_volatility': validate_numeric(option.get('impliedVolatility')),
                    'in_the_money': bool(option.get('inTheMoney')),
                    'contract_symbol': option.get('contractSymbol')
                }
                expiry_data['puts'].append(put_data)
            
            # Add summary statistics
            expiry_data['summary'] = {
                'total_calls': len(expiry_data['calls']),
                'total_puts': len(expiry_data['puts']),
                'call_volume_total': sum(call['volume'] for call in expiry_data['calls'] if call['volume'] is not None),
                'put_volume_total': sum(put['volume'] for put in expiry_data['puts'] if put['volume'] is not None),
                'max_call_open_interest': max((call['open_interest'] for call in expiry_data['calls'] if call['open_interest'] is not None), default=0),
                'max_put_open_interest': max((put['open_interest'] for put in expiry_data['puts'] if put['open_interest'] is not None), default=0)
            }
            
            options_data['expiration_dates'].append(expiry_data)
        
        # Add metadata
        options_data['metadata'] = {
            'last_updated': datetime.datetime.now().isoformat(),
            'data_source': 'Yahoo Finance',
            'expiration_dates_available': len(expiration_dates),
            'expiration_dates_returned': len(options_data['expiration_dates'])
        }
        
        return jsonify(options_data)
    
    except Exception as e:
        error_message = str(e)
        specific_error = "Data source error" if "request" in error_message.lower() else "Data processing error"
        return jsonify(error=f"Failed to fetch options data: {specific_error} - {error_message}"), 500
