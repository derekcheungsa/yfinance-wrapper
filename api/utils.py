import re
import time
from functools import wraps
from flask import request, jsonify
from concurrent.futures import ThreadPoolExecutor

def validate_ticker(ticker):
    """Validate ticker symbol format"""
    return bool(re.match(r'^[A-Za-z\.\-]{1,10}$', ticker))

def validate_tickers(tickers):
    """Validate an array of ticker symbols"""
    if not isinstance(tickers, list):
        return False
    if not tickers or len(tickers) > 10:  # Limit batch size to 10 tickers
        return False
    return all(validate_ticker(ticker) for ticker in tickers)

def process_tickers_parallel(tickers, process_function, max_workers=5):
    """Process multiple tickers in parallel"""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_ticker = {executor.submit(process_function, ticker): ticker for ticker in tickers}
        for future in future_to_ticker:
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                results[ticker] = {"error": str(e)}
    return results

class RateLimiter:
    def __init__(self, requests, window):
        self.requests = requests
        self.window = window
        self.clients = {}
    
    def limit(self, f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            
            if client_ip not in self.clients:
                self.clients[client_ip] = []
            
            now = time.time()
            
            # Remove old requests
            self.clients[client_ip] = [req_time for req_time in self.clients[client_ip]
                                     if now - req_time < self.window]
            
            # Check if rate limit is exceeded
            if len(self.clients[client_ip]) >= self.requests:
                return jsonify(error="Rate limit exceeded"), 429
            
            # Add new request
            self.clients[client_ip].append(now)
            
            return f(*args, **kwargs)
        return decorated_function
