from flask import Blueprint, jsonify, request
from api.extensions import cache
import yfinance as yf
from .utils import validate_ticker, RateLimiter
import datetime
import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def calculate_rating_distribution(recommendation_mean, num_analysts):
    """
    Calculate rating distribution based on recommendation mean and total analysts.
    recommendation_mean is on a scale of 1-5 where:
    1-1.5: Strong Buy
    1.5-2.5: Buy
    2.5-3.5: Hold
    3.5-4.5: Sell
    4.5-5: Strong Sell
    """
    if recommendation_mean is None or num_analysts is None or num_analysts == 0:
        return None

    # Calculate weights based on distance from recommendation_mean
    weights = {
        'strongBuy': max(0, min(1, (1.5 - recommendation_mean) / 0.5)),
        'buy': max(0, min(1, (2.5 - recommendation_mean) / 1.0 if recommendation_mean >= 1.5 else (recommendation_mean - 1.0) / 0.5)),
        'hold': max(0, min(1, (3.5 - recommendation_mean) / 1.0 if recommendation_mean >= 2.5 else (recommendation_mean - 1.5) / 1.0)),
        'sell': max(0, min(1, (4.5 - recommendation_mean) / 1.0 if recommendation_mean >= 3.5 else (recommendation_mean - 2.5) / 1.0)),
        'strongSell': max(0, min(1, (recommendation_mean - 4.5) / 0.5 if recommendation_mean >= 4.5 else 0))
    }

    # Calculate initial distribution
    total_weight = sum(weights.values())
    if total_weight == 0:
        return None

    distribution = {
        rating: round(weight * num_analysts / total_weight)
        for rating, weight in weights.items()
    }

    # Adjust to ensure total equals num_analysts
    total = sum(distribution.values())
    if total != num_analysts:
        # Find the rating with the highest weight and adjust it
        max_rating = max(weights.items(), key=lambda x: x[1])[0]
        distribution[max_rating] += (num_analysts - total)

    return distribution

def calculate_rating_trend(ticker, stock):
    try:
        recommendations = stock.recommendations
        if recommendations is None or recommendations.empty:
            return None

        trends = []
        # Convert period to datetime if it's a string
        if 'period' in recommendations.columns:
            recommendations['period'] = pd.to_datetime(recommendations['period'])

        for period, group in recommendations.groupby('period'):
            try:
                total_ratings = (
                    group['strongBuy'].iloc[0] +
                    group['buy'].iloc[0] +
                    group['hold'].iloc[0] +
                    group['sell'].iloc[0] +
                    group['strongSell'].iloc[0]
                )
                
                if total_ratings > 0:
                    weighted_rating = (
                        (1 * group['strongBuy'].iloc[0] +
                         2 * group['buy'].iloc[0] +
                         3 * group['hold'].iloc[0] +
                         4 * group['sell'].iloc[0] +
                         5 * group['strongSell'].iloc[0]) / total_ratings
                    )
                    
                    trend = {
                        'period': period.strftime('%Y-%m-%d') if hasattr(period, 'strftime') else str(period),
                        'weighted_rating': round(weighted_rating, 2),
                        'total_analysts': int(total_ratings),
                        'distribution': {
                            'strongBuy': int(group['strongBuy'].iloc[0]),
                            'buy': int(group['buy'].iloc[0]),
                            'hold': int(group['hold'].iloc[0]),
                            'sell': int(group['sell'].iloc[0]),
                            'strongSell': int(group['strongSell'].iloc[0])
                        }
                    }
                    trends.append(trend)
            except Exception as e:
                logger.error(f"Error processing period {period}: {str(e)}")
                continue
        
        return sorted(trends, key=lambda x: x['period'])
    except Exception as e:
        logger.error(f"Error calculating rating trends for {ticker}: {str(e)}")
        return None

@api_bp.route('/stock/insider_trades', methods=['POST'])
@rate_limiter.limit
@cache.cached(timeout=300)
def get_insider_trades():
    """Get insider trading data including institutional holders"""
    if not request.is_json:
        return jsonify(error="Request must be JSON"), 400

    request_data = request.get_json()
    if not request_data or 'ticker' not in request_data:
        return jsonify(error="Missing required field: ticker"), 400

    ticker = request_data['ticker']
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400

    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        shares_outstanding = validate_numeric(info.get('sharesOutstanding'), 0)
        
        # Get institutional holders
        institutional_holders = []
        inst_holders_df = stock.institutional_holders
        
        if inst_holders_df is not None and not inst_holders_df.empty:
            for _, holder in inst_holders_df.iterrows():
                holder_shares = validate_numeric(holder.get('Shares'), 0)
                # Calculate percentage of shares outstanding
                pct_out = round((holder_shares / shares_outstanding * 100), 2) if shares_outstanding > 0 else 0.0
                
                institutional_holders.append({
                    'holder': holder.get('Holder', ''),
                    'shares': holder_shares,
                    'pct_out': pct_out,
                    'value': validate_numeric(holder.get('Value'), 0),
                    'date_reported': holder.get('Date Reported', '').strftime('%Y-%m-%d') if pd.notnull(holder.get('Date Reported')) else None
                })

        # Get major holders data
        major_holders = []
        maj_holders_df = stock.major_holders
        
        if maj_holders_df is not None and not maj_holders_df.empty:
            for _, holder in maj_holders_df.iterrows():
                if len(holder) >= 2:  # Ensure we have both percentage and holder type
                    pct = validate_numeric(holder[0].strip('%'), 0) if holder[0] else 0.0
                    major_holders.append({
                        'pct_held': round(pct, 2),
                        'holder_type': holder[1]
                    })

        data = {
            'symbol': ticker,
            'shares_outstanding': shares_outstanding,
            'institutional_holders': institutional_holders,
            'major_holders': major_holders,
            'timestamp': datetime.datetime.now().isoformat()
        }

        return jsonify(data)

    except Exception as e:
        logger.error(f"Error fetching insider trades for {ticker}: {str(e)}")
        return jsonify(error=f"Failed to fetch insider trades data: {str(e)}"), 500

@api_bp.route('/stock/analyst_recommendations', methods=['POST'])
@rate_limiter.limit
@cache.cached(timeout=300)
def get_analyst_recommendations():
    """Get analyst recommendations for a stock"""
    if not request.is_json:
        return jsonify(error="Request must be JSON"), 400

    request_data = request.get_json()
    if not request_data or 'ticker' not in request_data:
        return jsonify(error="Missing required field: ticker"), 400

    ticker = request_data['ticker']
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400

    try:
        stock = yf.Ticker(ticker)
        recommendations = stock.recommendations
        
        if recommendations is None or recommendations.empty:
            return jsonify({
                'symbol': ticker,
                'recommendations': [],
                'timestamp': datetime.datetime.now().isoformat()
            })

        # Convert recommendations DataFrame to list of records
        recommendations_list = []
        for period, group in recommendations.groupby('period'):
            period_data = {
                'period': period,
                'strongBuy': int(group['strongBuy'].iloc[0]),
                'buy': int(group['buy'].iloc[0]),
                'hold': int(group['hold'].iloc[0]),
                'sell': int(group['sell'].iloc[0]),
                'strongSell': int(group['strongSell'].iloc[0])
            }
            recommendations_list.append(period_data)

        return jsonify({
            'symbol': ticker,
            'recommendations': recommendations_list,
            'timestamp': datetime.datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error fetching recommendations for {ticker}: {str(e)}")
        return jsonify(error=f"Failed to fetch analyst recommendations: {str(e)}"), 500

@api_bp.route('/stock/analyst_ratings', methods=['POST'])
@rate_limiter.limit
@cache.cached(timeout=300)
def get_analyst_ratings():
    """Get analyst ratings summary and trends for a stock"""
    if not request.is_json:
        return jsonify(error="Request must be JSON"), 400

    request_data = request.get_json()
    if not request_data or 'ticker' not in request_data:
        return jsonify(error="Missing required field: ticker"), 400

    ticker = request_data['ticker']
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Get current ratings
        recommendation_mean = validate_numeric(info.get('recommendationMean'))
        num_analysts = validate_numeric(info.get('numberOfAnalystOpinions'))
        distribution = calculate_rating_distribution(recommendation_mean, num_analysts)

        # Calculate rating trends
        trends = calculate_rating_trend(ticker, stock)

        data = {
            'symbol': ticker,
            'current_ratings': {
                'recommendation_mean': recommendation_mean,
                'num_analysts': num_analysts,
                'rating_distribution': distribution
            },
            'rating_trends': trends,
            'timestamp': datetime.datetime.now().isoformat()
        }

        return jsonify(data)

    except Exception as e:
        logger.error(f"Error fetching ratings for {ticker}: {str(e)}")
        return jsonify(error=f"Failed to fetch analyst ratings: {str(e)}"), 500

@api_bp.route('/stock/price_targets', methods=['POST'])
@rate_limiter.limit
@cache.cached(timeout=300)
def get_price_targets():
    """Get analyst price targets for a stock"""
    if not request.is_json:
        return jsonify(error="Request must be JSON"), 400

    request_data = request.get_json()
    if not request_data or 'ticker' not in request_data:
        return jsonify(error="Missing required field: ticker"), 400

    ticker = request_data['ticker']
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        data = {
            'symbol': ticker,
            'price_targets': {
                'high': validate_numeric(info.get('targetHighPrice')),
                'low': validate_numeric(info.get('targetLowPrice')),
                'mean': validate_numeric(info.get('targetMeanPrice')),
                'median': validate_numeric(info.get('targetMedianPrice'))
            },
            'num_analysts': validate_numeric(info.get('numberOfAnalystOpinions')),
            'timestamp': datetime.datetime.now().isoformat()
        }

        return jsonify(data)

    except Exception as e:
        logger.error(f"Error fetching price targets for {ticker}: {str(e)}")
        return jsonify(error=f"Failed to fetch price targets: {str(e)}"), 500

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
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close != 0 else 0.0

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
                'change_calculation': 'real_time' if info.get('regularMarketChange') is not None else 'calculated' if change is not None else 'unavailable'
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
                'date': index.strftime("%Y-%m-%d"),
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
                'market_cap': validate_numeric(info.get('marketCap')),
                'enterprise_value': validate_numeric(info.get('enterpriseValue')),
                'trailing_pe': validate_numeric(info.get('trailingPE')),
                'forward_pe': validate_numeric(info.get('forwardPE')),
                'profit_margins': validate_numeric(info.get('profitMargins')),
                'dividend_yield': validate_numeric(info.get('dividendYield')),
                'beta': validate_numeric(info.get('beta')),
                'fifty_two_week_high': validate_numeric(info.get('fiftyTwoWeekHigh')),
                'fifty_two_week_low': validate_numeric(info.get('fiftyTwoWeekLow'))
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
                'has_fundamental_data': any(info.get(key) is not None for key in ['trailingPE', 'forwardPE', 'profitMargins']),
                'has_detailed_info': all(info.get(key) is not None for key in ['longName', 'sector', 'industry'])
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
        return jsonify(error=f"Failed to fetch company information: {str(e)}"), 500

@api_bp.route('/stock/options', methods=['POST'])
@rate_limiter.limit
@cache.cached(timeout=300)
def get_options_data():
    """Get option chains for a given stock for the next 3 expiration dates"""
    if not request.is_json:
        return jsonify(error="Request must be JSON"), 400

    request_data = request.get_json()
    if not request_data or 'ticker' not in request_data:
        return jsonify(error="Missing required field: ticker"), 400

    ticker = request_data['ticker']
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400

    try:
        stock = yf.Ticker(ticker)
        
        # Get all expiration dates
        try:
            expirations = stock.options
        except:
            return jsonify(error="No options data available"), 404

        if not expirations:
            return jsonify(error="No options data available"), 404

        # Take the next 3 expiration dates
        next_expirations = expirations[:3]

        # Get option chain data for each expiration
        options_data = {}
        for date in next_expirations:
            try:
                opt = stock.option_chain(date)
                
                # Process calls
                calls = [{
                    'strike': validate_numeric(row['strike']),
                    'lastPrice': validate_numeric(row['lastPrice']),
                    'bid': validate_numeric(row['bid']),
                    'ask': validate_numeric(row['ask']),
                    'volume': validate_numeric(row['volume']),
                    'openInterest': validate_numeric(row['openInterest']),
                    'impliedVolatility': validate_numeric(row['impliedVolatility'])
                } for _, row in opt.calls.iterrows()]

                # Process puts
                puts = [{
                    'strike': validate_numeric(row['strike']),
                    'lastPrice': validate_numeric(row['lastPrice']),
                    'bid': validate_numeric(row['bid']),
                    'ask': validate_numeric(row['ask']),
                    'volume': validate_numeric(row['volume']),
                    'openInterest': validate_numeric(row['openInterest']),
                    'impliedVolatility': validate_numeric(row['impliedVolatility'])
                } for _, row in opt.puts.iterrows()]

                options_data[date] = {
                    'calls': calls,
                    'puts': puts
                }
            except Exception as e:
                logger.error(f"Error fetching options for date {date}: {str(e)}")
                continue

        return jsonify({
            'symbol': ticker,
            'options_data': options_data,
            'timestamp': datetime.datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error fetching options data for {ticker}: {str(e)}")
        return jsonify(error=f"Failed to fetch options data: {str(e)}"), 500
