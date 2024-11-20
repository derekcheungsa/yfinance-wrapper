from flask import Blueprint, jsonify, request
from api.extensions import cache
import yfinance as yf
from .utils import validate_ticker, RateLimiter, validate_tickers, process_tickers_parallel
import datetime
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor

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
        change_percent = validate_numeric(info.get('regularMarketChangePercent'))

        # If change values are None, try calculating from current and previous prices
        if (change is None or change_percent is None) and current_price is not None and previous_close is not None:
            change, change_percent = calculate_change_values(current_price, previous_close)

        # If still None, try alternative fields
        if change is None:
            change = validate_numeric(info.get('regularMarketPrice')) - validate_numeric(info.get('regularMarketPreviousClose'))

        # If still None, try historical data
        if (change is None or change_percent is None) and current_price is not None:
            hist_change, hist_change_percent = get_historical_changes(stock, ticker)
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
                'date': index.strftime("%Y-%m-%d"),  # Change here to format date
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
def get_option_chain():
    """Get option chains for a given stock for the next 3 expiration dates"""
    # Validate request body
    if not request.is_json:
        return jsonify(error="Request must be JSON"), 400

    request_data = request.get_json()
    
    # Validate required fields
    if not request_data or 'ticker' not in request_data:
        return jsonify(error="Missing required field: ticker"), 400

    ticker = request_data['ticker']
    
    # Validate ticker
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400

    try:
        stock = yf.Ticker(ticker)
        expiration_dates = stock.options[:3]  # Fetch next 3 expiration dates

        if not expiration_dates:
            return jsonify(error="No options data available for this ticker"), 404

        option_data = {
            'symbol': ticker,
            'expirations': {}
        }

        for date in expiration_dates:
            try:
                options = stock.option_chain(date)
                
                # Validate options data
                if options is None or not hasattr(options, 'calls') or not hasattr(options, 'puts'):
                    logger.warning(f"Invalid options data structure for {ticker} on {date}")
                    continue

                # Replace NaN values with None and convert numeric columns
                def clean_options_data(df):
                    # Replace infinite values with NaN first
                    df = df.replace([np.inf, -np.inf], np.nan)
                    
                    # Replace NaN with specific values based on column type
                    numeric_cols = ['strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 
                                'volume', 'openInterest', 'impliedVolatility']
                    
                    # For numeric columns, replace NaN with 0
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = df[col].fillna(0)
                    
                    # For boolean columns like inTheMoney, replace NaN with False
                    if 'inTheMoney' in df.columns:
                        df['inTheMoney'] = df['inTheMoney'].fillna(False)
                    
                    # Convert to proper types
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = df[col].astype(float)
                    
                    return df

                # Clean and convert DataFrames to records
                try:
                    calls = clean_options_data(options.calls).to_dict(orient='records')
                    puts = clean_options_data(options.puts).to_dict(orient='records')

                    option_data['expirations'][date] = {
                        'call_options': calls,
                        'put_options': puts
                    }
                except Exception as e:
                    logger.error(f"Error processing options for date {date}: {str(e)}")
                    continue

                if not option_data['expirations']:
                    return jsonify(error="No valid options data available for this ticker"), 404

                option_data['timestamp'] = datetime.datetime.now().isoformat()
                return jsonify(option_data)

            except Exception as e:
                logger.error(f"Error processing options for date {date}: {str(e)}")
                continue

        if not option_data['expirations']:
            return jsonify(error="No valid options data available for this ticker"), 404

        option_data['timestamp'] = datetime.datetime.now().isoformat()
        return jsonify(option_data)

    except Exception as e:
        logger.error(f"Failed to fetch option chains for {ticker}: {str(e)}")
        return jsonify(error=f"Failed to fetch option chains: {str(e)}"), 500


@api_bp.route('/stock/<ticker>/earnings_estimate')
@rate_limiter.limit
@cache.cached(timeout=300)
def get_earnings_estimate(ticker):
    """Get earnings estimate data for a specific ticker"""
    if not validate_ticker(ticker):
        return jsonify(error="Invalid ticker symbol"), 400

    try:
        stock = yf.Ticker(ticker)
        earnings_estimates = stock.earnings_estimate

        if earnings_estimates is not None:
            # Adding 'Quarter' information to the earnings estimates output
            earnings_estimates['Quarter'] = earnings_estimates.index.to_list()

            data = {
                'symbol': ticker,
                'earnings_estimates': earnings_estimates.to_dict('records')  # Convert DataFrame to list of records
            }
        else:
            data = {'symbol': ticker, 'earnings_estimates': None}

        return jsonify(data)
    except Exception as e:
        return jsonify(error=f"Failed to fetch earnings estimate data: {str(e)}"), 500


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


def validate_tickers(tickers):
    """Validate and sanitize a list of tickers"""
    if not isinstance(tickers, list):
        return False
    
    if len(tickers) > 10:
        return False
    
    return all(validate_ticker(ticker) for ticker in tickers)

def process_tickers_parallel(tickers, func):
    """Process a list of tickers in parallel using a thread pool"""
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(func, tickers))
    return results


@api_bp.route('/stock/analyst_ratings', methods=['POST'])
@rate_limiter.limit
@cache.cached(timeout=300)
def get_analyst_ratings():
    """Get analyst ratings summary for given stocks"""
    if not request.is_json:
        return jsonify(error="Request must be JSON"), 400

    request_data = request.get_json()
    
    if not request_data:
        return jsonify(error="Missing request body"), 400

    # Handle both single ticker and multiple tickers
    tickers = request_data.get('tickers', [request_data.get('ticker')])
    
    if not validate_tickers(tickers):
        return jsonify(error="Invalid ticker symbol(s). Maximum 10 tickers allowed."), 400

    def process_single_ticker(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Extract key metrics
            recommendation_mean = validate_numeric(info.get('recommendationMean'))
            num_analysts = validate_numeric(info.get('numberOfAnalystOpinions'))

            # Calculate rating distribution
            distribution = calculate_rating_distribution(recommendation_mean, num_analysts)

            return {
                'symbol': ticker,
                'average_rating': recommendation_mean,
                'num_analysts': num_analysts,
                'distribution': distribution,
            }
        except Exception as e:
            logger.error(f"Error processing analyst ratings for {ticker}: {str(e)}")
            return {
                'symbol': ticker,
                'error': str(e)
            }

    # Process all tickers
    results = process_tickers_parallel(tickers, process_single_ticker)

    # Format response
    response = {
        'ratings': [result for result in results if result is not None],
        'timestamp': datetime.datetime.now().isoformat()
    }

    return jsonify(response)

@api_bp.route('/stock/analyst_recommendations', methods=['POST'])
@rate_limiter.limit
@cache.cached(timeout=300)
def get_analyst_recommendations():
    """Get analyst recommendations for given stocks"""
    if not request.is_json:
        return jsonify(error="Request must be JSON"), 400

    request_data = request.get_json()
    if not request_data:
        return jsonify(error="Missing request body"), 400

    # Handle both single ticker and multiple tickers
    tickers = request_data.get('tickers', [request_data.get('ticker')])
    
    if not validate_tickers(tickers):
        return jsonify(error="Invalid ticker symbol(s). Maximum 10 tickers allowed."), 400

    def process_single_ticker(ticker):
        try:
            stock = yf.Ticker(ticker)
            recommendations = stock.recommendations
            
            logger.info(f"Raw recommendations data for {ticker}: {recommendations}")
            
            if recommendations is None or recommendations.empty:
                return {
                    'symbol': ticker,
                    'recommendations': [],
                }

            recommendations_list = []
            for idx, row in recommendations.iterrows():
                try:
                    rec = {
                        'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx),
                        'firm': row.get('Firm', None),
                        'to_grade': row.get('To Grade', None),
                        'from_grade': row.get('From Grade', None),
                        'action': row.get('Action', None)
                    }
                    recommendations_list.append(rec)
                except Exception as e:
                    logger.error(f"Error processing recommendation row for {ticker}: {e}")
                    continue

            return {
                'symbol': ticker,
                'recommendations': recommendations_list,
            }

        except Exception as e:
            logger.error(f"Error processing analyst recommendations for {ticker}: {str(e)}")
            return {
                'symbol': ticker,
                'error': str(e)
            }

    # Process all tickers
    results = process_tickers_parallel(tickers, process_single_ticker)

    # Format response
    response = {
        'recommendations': [result for result in results if result is not None],
        'timestamp': datetime.datetime.now().isoformat()
    }

    return jsonify(response)

@api_bp.route('/stock/price_targets', methods=['POST'])
@rate_limiter.limit
@cache.cached(timeout=300)
def get_price_targets():
    """Get price targets for given stocks"""
    if not request.is_json:
        return jsonify(error="Request must be JSON"), 400

    request_data = request.get_json()
    if not request_data:
        return jsonify(error="Missing request body"), 400

    # Handle both single ticker and multiple tickers
    tickers = request_data.get('tickers', [request_data.get('ticker')])
    
    if not validate_tickers(tickers):
        return jsonify(error="Invalid ticker symbol(s). Maximum 10 tickers allowed."), 400

    def process_single_ticker(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            return {
                'symbol': ticker,
                'target_high': validate_numeric(info.get('targetHighPrice')),
                'target_low': validate_numeric(info.get('targetLowPrice')),
                'target_mean': validate_numeric(info.get('targetMeanPrice')),
                'target_median': validate_numeric(info.get('targetMedianPrice')),
                'num_analysts': validate_numeric(info.get('numberOfAnalystOpinions'))
            }

        except Exception as e:
            logger.error(f"Error processing price targets for {ticker}: {str(e)}")
            return {
                'symbol': ticker,
                'error': str(e)
            }

    # Process all tickers
    results = process_tickers_parallel(tickers, process_single_ticker)

    # Format response
    response = {
        'price_targets': [result for result in results if result is not None],
        'timestamp': datetime.datetime.now().isoformat()
    }

    return jsonify(response)