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

def calculate_earnings_growth(stock):
    """Calculate earnings growth from historical data if not available"""
    try:
        earnings = stock.earnings
        if earnings is not None and not earnings.empty and len(earnings) >= 2:
            latest = earnings.iloc[-1]
            previous = earnings.iloc[-2]
            if previous != 0:
                return float((latest - previous) / abs(previous))
    except Exception:
        pass
    return None

def calculate_revenue_growth(stock):
    """Calculate revenue growth from historical data if not available"""
    try:
        financials = stock.financials
        if financials is not None and not financials.empty and 'Total Revenue' in financials.index:
            revenues = financials.loc['Total Revenue']
            if len(revenues) >= 2:
                latest = revenues.iloc[0]
                previous = revenues.iloc[1]
                if previous != 0:
                    return float((latest - previous) / abs(previous))
    except Exception:
        pass
    return None

@api_bp.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="Rate limit exceeded. Please try again later."), 429

@api_bp.errorhandler(400)
def bad_request_handler(e):
    return jsonify(error=str(e)), 400

@api_bp.route('/stock/<ticker>/analysis/earnings')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_earnings_estimates(ticker):
    """Get earnings (EPS) estimates"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Calculate fallback values if primary data is missing
        earnings_growth = validate_numeric(info.get('earningsGrowth'))
        if earnings_growth is None:
            earnings_growth = calculate_earnings_growth(stock)
        
        earnings_data = {
            'symbol': ticker,
            'estimates': {
                'current_year_estimate': validate_numeric(info.get('epsCurrentYear')),
                'next_quarter_estimate': validate_numeric(info.get('epsNextQuarter')),
                'next_year_estimate': validate_numeric(info.get('epsForward')),
                'trailing_eps': validate_numeric(info.get('trailingEps')),
                'forward_eps': validate_numeric(info.get('forwardEps'))
            },
            'growth_metrics': {
                'earnings_growth': earnings_growth,
                'earnings_quarterly_growth': validate_numeric(info.get('earningsQuarterlyGrowth'))
            },
            'data_quality': {
                'has_estimates': any(x is not None for x in [
                    info.get('epsCurrentYear'),
                    info.get('epsNextQuarter'),
                    info.get('epsForward')
                ]),
                'has_historical': any(x is not None for x in [
                    info.get('trailingEps'),
                    info.get('earningsGrowth')
                ])
            }
        }
        
        return jsonify(earnings_data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch earnings data: {str(e)}"), 500

@api_bp.route('/stock/<ticker>/analysis/revenue')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_revenue_forecasts(ticker):
    """Get revenue forecasts"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Calculate fallback values if primary data is missing
        revenue_growth = validate_numeric(info.get('revenueGrowth'))
        if revenue_growth is None:
            revenue_growth = calculate_revenue_growth(stock)
        
        revenue_data = {
            'symbol': ticker,
            'metrics': {
                'revenue_growth': revenue_growth,
                'revenue_per_share': validate_numeric(info.get('revenuePerShare')),
                'trailing_revenue': validate_numeric(info.get('totalRevenue')),
                'quarterly_revenue_growth': validate_numeric(info.get('revenueQuarterlyGrowth')),
                'forward_revenue': validate_numeric(info.get('forwardRevenue')),
                'profit_margins': validate_numeric(info.get('profitMargins'))
            },
            'data_quality': {
                'has_growth_metrics': any(x is not None for x in [
                    revenue_growth,
                    info.get('revenueQuarterlyGrowth')
                ]),
                'has_current_metrics': any(x is not None for x in [
                    info.get('totalRevenue'),
                    info.get('revenuePerShare')
                ])
            }
        }
        
        return jsonify(revenue_data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch revenue data: {str(e)}"), 500

@api_bp.route('/stock/<ticker>/analysis/recommendations')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_analyst_recommendations(ticker):
    """Get analyst recommendations"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        info = stock.info
        
        # Process recommendations data
        recommendations_list = []
        if recommendations is not None and not recommendations.empty:
            # Get the last 90 days of recommendations
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=90)
            recent_recommendations = recommendations[recommendations.index >= cutoff_date]
            
            recommendations_list = [
                {
                    'date': index.strftime('%Y-%m-%d'),
                    'firm': str(row.get('Firm', '')),
                    'to_grade': str(row.get('To Grade', '')),
                    'from_grade': str(row.get('From Grade', '')),
                    'action': str(row.get('Action', ''))
                }
                for index, row in recent_recommendations.iterrows()
            ]
        
        response_data = {
            'symbol': ticker,
            'summary': {
                'recommendation_mean': validate_numeric(info.get('recommendationMean')),
                'recommendation_key': info.get('recommendationKey', 'N/A'),
                'number_of_analysts': validate_numeric(info.get('numberOfAnalystOpinions'), 0),
                'recommendations_count': len(recommendations_list)
            },
            'recommendations': recommendations_list,
            'data_quality': {
                'has_recent_recommendations': len(recommendations_list) > 0,
                'has_analyst_ratings': info.get('recommendationMean') is not None
            }
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch analyst recommendations: {str(e)}"), 500

@api_bp.route('/stock/<ticker>/analysis/price-targets')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_price_targets(ticker):
    """Get analyst price targets"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400
    
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        current_price = validate_numeric(info.get('currentPrice'))
        if current_price is None:
            # Fallback to regularMarketPrice if currentPrice is not available
            current_price = validate_numeric(info.get('regularMarketPrice'))
        
        target_mean_price = validate_numeric(info.get('targetMeanPrice'))
        target_median_price = validate_numeric(info.get('targetMedianPrice'))
        
        # Calculate implied change if we have both current price and target
        implied_change = None
        if current_price is not None and target_mean_price is not None:
            implied_change = ((target_mean_price - current_price) / current_price) * 100
        
        price_target_data = {
            'symbol': ticker,
            'current_price': current_price,
            'targets': {
                'high_price': validate_numeric(info.get('targetHighPrice')),
                'low_price': validate_numeric(info.get('targetLowPrice')),
                'mean_price': target_mean_price,
                'median_price': target_median_price
            },
            'analysis': {
                'number_of_analysts': validate_numeric(info.get('numberOfAnalystOpinions'), 0),
                'implied_change_percent': validate_numeric(implied_change),
                'recommendation_key': info.get('recommendationKey', 'N/A'),
                'recommendation_mean': validate_numeric(info.get('recommendationMean'))
            },
            'data_quality': {
                'has_price_targets': any(x is not None for x in [
                    info.get('targetHighPrice'),
                    info.get('targetLowPrice'),
                    info.get('targetMeanPrice')
                ]),
                'has_current_price': current_price is not None,
                'has_analyst_ratings': info.get('recommendationMean') is not None
            }
        }
        
        return jsonify(price_target_data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch price targets: {str(e)}"), 500
