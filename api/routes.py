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
        
        # Validate data source
        if not info:
            return jsonify(error="Failed to fetch stock data from Yahoo Finance"), 500
        
        # Get earnings estimates data with fallbacks
        eps_current = validate_numeric(info.get('trailingEps'))
        eps_forward = validate_numeric(info.get('forwardEps'))
        eps_current_year = validate_numeric(info.get('currentYearEps'))
        eps_next_quarter = validate_numeric(info.get('earningsEstimate', {}).get('nextQuarter'))
        eps_next_year = validate_numeric(info.get('earningsNextYear'))
        
        # Get revenue estimates
        revenue_current = validate_numeric(info.get('totalRevenue'))
        revenue_estimate = validate_numeric(info.get('revenueEstimate', {}).get('avg'))
        revenue_growth = validate_numeric(info.get('revenueGrowth'))
        
        # Calculate growth rates with None handling
        eps_growth = None
        if eps_forward is not None and eps_current is not None and eps_current != 0:
            eps_growth = ((eps_forward - eps_current) / abs(eps_current)) * 100
            
        revenue_growth_estimate = None
        if revenue_estimate is not None and revenue_current is not None and revenue_current != 0:
            revenue_growth_estimate = ((revenue_estimate - revenue_current) / revenue_current) * 100
            
        # Get analyst coverage data
        earnings_analysts = info.get('earningsEstimate', {}).get('numberOfAnalysts')
        revenue_analysts = info.get('revenueEstimate', {}).get('numberOfAnalysts')
        
        # Get earnings calendar data with validation
        calendar = stock.calendar
        earnings_dates = {
            'last': None,
            'next': None
        }
        
        if calendar is not None and isinstance(calendar, pd.DataFrame):
            try:
                earnings_date = calendar.get('Earnings Date')
                if isinstance(earnings_date, pd.Series) and not earnings_date.empty:
                    earnings_dates['last'] = earnings_date.iloc[0].strftime('%Y-%m-%d') if earnings_date.iloc[0] else None
                    earnings_dates['next'] = earnings_date.iloc[-1].strftime('%Y-%m-%d') if len(earnings_date) > 1 and earnings_date.iloc[-1] else None
            except (AttributeError, IndexError) as e:
                print(f"Error processing calendar data: {str(e)}")
        
        # Implement fallback calculations for missing estimates
        if eps_next_quarter is None and eps_current is not None and eps_growth is not None:
            eps_next_quarter = eps_current * (1 + eps_growth/100)
        
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
            'revenue': {
                'current': revenue_current,
                'estimate': revenue_estimate,
                'growth_current': validate_numeric(revenue_growth),
                'growth_estimate': validate_numeric(revenue_growth_estimate)
            },
            'ratios': {
                'pe_ratio': validate_numeric(info.get('trailingPE')),
                'forward_pe': validate_numeric(info.get('forwardPE')),
                'peg_ratio': validate_numeric(info.get('pegRatio'))
            },
            'coverage': {
                'earnings_analysts': earnings_analysts,
                'revenue_analysts': revenue_analysts,
                'total_analysts': validate_numeric(info.get('numberOfAnalystOpinions'))
            },
            'earnings_dates': earnings_dates,
            'data_quality': {
                'eps_estimates': {
                    'has_current': eps_current is not None,
                    'has_forward': eps_forward is not None,
                    'has_next_quarter': eps_next_quarter is not None,
                    'has_growth': eps_growth is not None
                },
                'revenue_estimates': {
                    'has_current': revenue_current is not None,
                    'has_estimate': revenue_estimate is not None,
                    'has_growth': revenue_growth is not None
                },
                'coverage_quality': {
                    'has_earnings_analysts': earnings_analysts is not None,
                    'has_revenue_analysts': revenue_analysts is not None
                },
                'dates_quality': {
                    'has_next_date': earnings_dates['next'] is not None,
                    'has_last_date': earnings_dates['last'] is not None
                }
            },
            'metadata': {
                'last_updated': datetime.datetime.now().isoformat(),
                'data_source': 'Yahoo Finance'
            }
        }
        
        return jsonify(earnings_data)
    except Exception as e:
        error_message = str(e)
        specific_error = "Data source error" if "request" in error_message.lower() else "Data processing error"
        return jsonify(error=f"Failed to fetch earnings data: {specific_error} - {error_message}"), 500