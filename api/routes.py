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

# ... [previous endpoints remain unchanged until line 264] ...

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
        
        # Get earnings estimates
        earnings_data = {
            'symbol': ticker,
            'current_year_estimate': info.get('epsCurrentYear'),
            'next_quarter_estimate': info.get('epsNextQuarter'),
            'next_year_estimate': info.get('epsForward'),
            'trailing_eps': info.get('trailingEps'),
            'forward_eps': info.get('forwardEps'),
            'earnings_growth': info.get('earningsGrowth'),
            'earnings_quarterly_growth': info.get('earningsQuarterlyGrowth')
        }
        
        return jsonify(earnings_data)
    except Exception as e:
        return jsonify(error=str(e)), 500

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
        
        # Get revenue data
        revenue_data = {
            'symbol': ticker,
            'revenue_growth': info.get('revenueGrowth'),
            'revenue_per_share': info.get('revenuePerShare'),
            'trailing_revenue': info.get('totalRevenue'),
            'quarterly_revenue_growth': info.get('revenueQuarterlyGrowth'),
            'forward_revenue': info.get('forwardRevenue'),
            'profit_margins': info.get('profitMargins')
        }
        
        return jsonify(revenue_data)
    except Exception as e:
        return jsonify(error=str(e)), 500

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
        
        if recommendations is not None and not recommendations.empty:
            # Convert recommendations DataFrame to list of dictionaries
            recommendations_list = [
                {
                    'date': index.strftime('%Y-%m-%d'),
                    'firm': row.get('Firm', ''),
                    'to_grade': row.get('To Grade', ''),
                    'from_grade': row.get('From Grade', ''),
                    'action': row.get('Action', '')
                }
                for index, row in recommendations.iterrows()
            ]
        else:
            recommendations_list = []
        
        return jsonify({
            'symbol': ticker,
            'recommendations': recommendations_list
        })
    except Exception as e:
        return jsonify(error=str(e)), 500

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
        
        # Get price target data
        price_target_data = {
            'symbol': ticker,
            'current_price': info.get('currentPrice'),
            'target_high_price': info.get('targetHighPrice'),
            'target_low_price': info.get('targetLowPrice'),
            'target_mean_price': info.get('targetMeanPrice'),
            'target_median_price': info.get('targetMedianPrice'),
            'number_of_analysts': info.get('numberOfAnalystOpinions'),
            'recommendation_key': info.get('recommendationKey'),
            'recommendation_mean': info.get('recommendationMean')
        }
        
        return jsonify(price_target_data)
    except Exception as e:
        return jsonify(error=str(e)), 500
