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
        change_percent = validate_numeric(
            info.get('regularMarketChangePercent'))

        # If change values are None, try calculating from current and previous prices
        if (change is None or change_percent is None
            ) and current_price is not None and previous_close is not None:
            change, change_percent = calculate_change_values(
                current_price, previous_close)

        # If still None, try alternative fields
        if change is None:
            change = validate_numeric(
                info.get('regularMarketPrice')) - validate_numeric(
                    info.get('regularMarketPreviousClose'))

        # If still None, try historical data
        if (change is None
                or change_percent is None) and current_price is not None:
            hist_change, hist_change_percent = get_historical_changes(
                stock, ticker)
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
                'price_source':
                'real_time' if current_price is not None else 'unavailable',
                'change_calculation':
                'real_time' if info.get('regularMarketChange') is not None else
                'calculated' if change is not None else 'unavailable'
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
            'symbol':
            ticker,
            'period':
            period,
            'interval':
            interval,
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


@api_bp.route('/stock/<ticker>/info')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_company_info(ticker):
    """Get detailed company information"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Basic company information
        data = {
            'symbol': ticker,
            'company_name': info.get('longName'),
            'description': info.get('longBusinessSummary'),
            'industry': info.get('industry'),
            'sector': info.get('sector'),
            'website': info.get('website'),

            # Location and contact
            'address': {
                'city': info.get('city'),
                'state': info.get('state'),
                'country': info.get('country'),
                'zip': info.get('zip'),
                'phone': info.get('phone')
            },

            # Financial metrics
            'financial_metrics': {
                'market_cap':
                validate_numeric(info.get('marketCap')),
                'enterprise_value':
                validate_numeric(info.get('enterpriseValue')),
                'trailing_pe':
                validate_numeric(info.get('trailingPE')),
                'forward_pe':
                validate_numeric(info.get('forwardPE')),
                'profit_margins':
                validate_numeric(info.get('profitMargins')),
                'dividend_yield':
                validate_numeric(info.get('dividendYield')),
                'beta':
                validate_numeric(info.get('beta')),
                'fifty_two_week_high':
                validate_numeric(info.get('fiftyTwoWeekHigh')),
                'fifty_two_week_low':
                validate_numeric(info.get('fiftyTwoWeekLow'))
            },

            # Company officers
            'officers': info.get('companyOfficers', []),

            # Additional information
            'employees': info.get('fullTimeEmployees'),
            'founded_year': info.get('foundedYear'),
            'exchange': info.get('exchange'),
            'currency': info.get('currency'),

            # Data quality indicators
            'data_quality': {
                'has_fundamental_data':
                any(
                    info.get(key) is not None
                    for key in ['trailingPE', 'forwardPE', 'profitMargins']),
                'has_detailed_info':
                all(
                    info.get(key) is not None
                    for key in ['longName', 'sector', 'industry'])
            }
        }

        # Clean up officers data to include only relevant fields
        if data['officers']:
            data['officers'] = [{
                'name': officer.get('name'),
                'title': officer.get('title'),
                'year_born': officer.get('yearBorn')
            } for officer in data['officers']]

        return jsonify(data)
    except Exception as e:
        return jsonify(
            error=f"Failed to fetch company information: {str(e)}"), 500


@api_bp.route('/stock/<ticker>/options')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_option_chain(ticker):
    """Get option chains for a given stock"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400

    try:
        stock = yf.Ticker(ticker)
        options = stock.option_chain()

        option_data = {
            'symbol': ticker,
            'call_options': options.calls.to_dict(orient='records'),
            'put_options': options.puts.to_dict(orient='records'),
            'timestamp': datetime.datetime.now().isoformat()
        }

        return jsonify(option_data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch option chains: {str(e)}"), 500


@api_bp.route('/stock/<ticker>/eps_trend')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_eps_trend(ticker):
    """Get EPS trend for a stock"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400

    try:
        stock = yf.Ticker(ticker)
        eps_trend = stock.earnings_trend

        if not eps_trend:
            return jsonify(error="Could not retrieve EPS trend."), 404

        return jsonify({'symbol': ticker, 'eps_trend': eps_trend})
    except Exception as e:
        return jsonify(error=f"Failed to fetch EPS trend: {str(e)}"), 500
