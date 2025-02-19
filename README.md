# Yahoo Finance API Wrapper

A REST API wrapper for Yahoo Finance data built with Flask, providing real-time stock data, historical prices, company information, and earnings data. This is a wrapper on the popular yfinance python library.


## Features

- Real-time stock data endpoints
- Historical price data with customizable periods
- Detailed company information
- Options chain data
- Earnings estimates data
- Analyst recommendations and ratings

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
python main.py
```

The server will start at `http://0.0.0.0:5000`

## API Endpoints

### Stock Data
- `GET /api/stock/{ticker}` - Get current stock data
- `GET /api/stock/{ticker}/history` - Get historical stock data
- `GET /api/stock/{ticker}/info` - Get company information

### Options Data
- `POST /api/stock/options` - Get options chain data
  - Request body: `{ "ticker": "AAPL" }`

### Earnings Data
- `GET /api/stock/{ticker}/earnings_estimate` - Get earnings estimates

### Analyst Data
- `POST /api/stock/analyst_recommendations` - Get analyst recommendations history
- `POST /api/stock/analyst_ratings` - Get current analyst ratings summary
- `POST /api/stock/price_targets` - Get analyst price targets

## Query Parameters

- `period` (optional): Time period for historical data (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
- `interval` (optional): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)

## Rate Limiting

The API implements rate limiting of 100 requests per minute per IP address.


## Example Usage

```python
import requests

# Get current stock data
response = requests.get('http://localhost:5000/api/stock/AAPL')
stock_data = response.json()

# Get options data
response = requests.post('http://localhost:5000/api/stock/options', json={"ticker": "AAPL"})
options_data = response.json()

# Get historical data with parameters
params = {'period': '1y', 'interval': '1d'}
response = requests.get('http://localhost:5000/api/stock/AAPL/history', params=params)
historical_data = response.json()

# Get analyst ratings
response = requests.post('http://localhost:5000/api/stock/analyst_ratings', 
    json={"ticker": "AAPL"})
ratings_data = response.json()
```

## Error Handling

The API returns appropriate HTTP status codes:
- 200: Successful request
- 400: Invalid input (e.g., invalid ticker, missing required fields)
- 429: Rate limit exceeded
- 500: Server error

## Development

Built with:
- Flask (Python web framework)
- yfinance (Yahoo Finance API)
- pandas & numpy (Data processing)
- Flask-Caching (Response caching)

## Support
If you find this project helpful and would like to support my work, consider buying me a coffee at [buymeacoffee.com/aifornoncoders](https://buymeacoffee.com/aifornoncoders). Your support means a lot!

## Disclaimer

Yahoo!, Y!Finance, and Yahoo! finance are registered trademarks of Yahoo, Inc.

yfinance is not affiliated, endorsed, or vetted by Yahoo, Inc. It's an open-source tool that uses Yahoo's publicly available APIs, and is intended for research and educational purposes.

You should refer to Yahoo!'s terms of use for details on your rights to use the actual data downloaded. Remember - the Yahoo! finance API is intended for personal use only
