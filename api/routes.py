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

@api_bp.route('/stock/<ticker>/analysis/price-targets')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_price_targets(ticker):
    """Get analyst price targets for a stock"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get current price for reference
        current_price = validate_numeric(info.get('currentPrice'))
        if current_price is None:
            current_price = validate_numeric(info.get('regularMarketPrice'))
        
        # Get price target data with fallbacks
        target_mean = validate_numeric(info.get('targetMeanPrice'))
        target_median = validate_numeric(info.get('targetMedianPrice'))
        target_high = validate_numeric(info.get('targetHighPrice'))
        target_low = validate_numeric(info.get('targetLowPrice'))
        
        # Calculate potential returns if we have valid targets and current price
        mean_return_percent = None
        if target_mean is not None and current_price is not None and current_price != 0:
            mean_return_percent = ((target_mean - current_price) / current_price) * 100
        
        # Get number of analysts
        num_analysts = info.get('numberOfAnalystOpinions')
        if isinstance(num_analysts, str):
            try:
                num_analysts = int(num_analysts)
            except ValueError:
                num_analysts = None
        
        price_target_data = {
            'symbol': ticker,
            'current_price': current_price,
            'price_targets': {
                'mean': target_mean,
                'median': target_median,
                'high': target_high,
                'low': target_low,
                'mean_return_percent': validate_numeric(mean_return_percent)
            },
            'coverage': {
                'number_of_analysts': num_analysts,
                'last_update': info.get('lastPriceTargetUpdate')
            },
            'data_quality': {
                'has_current_price': current_price is not None,
                'has_mean_target': target_mean is not None,
                'has_analyst_count': num_analysts is not None,
                'target_spread': validate_numeric(target_high - target_low) if target_high is not None and target_low is not None else None
            }
        }
        
        return jsonify(price_target_data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch price target data: {str(e)}"), 500

@api_bp.route('/stock/<ticker>/analysis/earnings')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_earnings_estimates(ticker):
    """Get EPS estimates and earnings data"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Get earnings estimates data
        eps_current = validate_numeric(info.get('trailingEps'))
        eps_forward = validate_numeric(info.get('forwardEps'))
        eps_current_year = validate_numeric(info.get('currentYearEps'))
        eps_next_quarter = validate_numeric(info.get('earningsQuarterlyGrowth'))  # Fixed: Using correct field for quarterly estimates
        eps_next_year = validate_numeric(info.get('earningsNextYear'))
        
        # Calculate growth rates
        eps_growth = None
        if eps_forward is not None and eps_current is not None and eps_current != 0:
            eps_growth = ((eps_forward - eps_current) / abs(eps_current)) * 100
            
        # Get earnings calendar data
        calendar = stock.calendar
        if calendar is not None:
            last_earnings_date = calendar.get('Earnings Date', [None])[0] if isinstance(calendar.get('Earnings Date'), list) else None
            next_earnings_date = calendar.get('Earnings Date', [None])[-1] if isinstance(calendar.get('Earnings Date'), list) else None
        else:
            last_earnings_date = None
            next_earnings_date = None
            
        earnings_data = {
            'symbol': ticker,
            'eps': {
                'current': eps_current,
                'forward': eps_forward,
                'current_year': eps_current_year,
                'next_quarter': eps_next_quarter,
                'next_year': eps_next_year,
                'growth_percent': validate_numeric(eps_growth)
            },
            'ratios': {
                'pe_ratio': validate_numeric(info.get('trailingPE')),
                'forward_pe': validate_numeric(info.get('forwardPE')),
                'peg_ratio': validate_numeric(info.get('pegRatio'))
            },
            'earnings_dates': {
                'last_earnings_date': last_earnings_date.strftime('%Y-%m-%d') if last_earnings_date else None,
                'next_earnings_date': next_earnings_date.strftime('%Y-%m-%d') if next_earnings_date else None
            },
            'data_quality': {
                'has_current_eps': eps_current is not None,
                'has_forward_eps': eps_forward is not None,
                'has_pe_ratio': info.get('trailingPE') is not None,
                'has_earnings_dates': next_earnings_date is not None,
                'has_growth_estimates': eps_growth is not None
            }
        }
        
        return jsonify(earnings_data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch earnings data: {str(e)}"), 500