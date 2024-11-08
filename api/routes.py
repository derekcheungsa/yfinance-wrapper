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
