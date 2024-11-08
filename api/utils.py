import re
import time
from functools import wraps
from flask import request, jsonify

def validate_ticker(ticker):
    """Validate ticker symbol format"""
    return bool(re.match(r'^[A-Za-z\.\-]{1,10}$', ticker))

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
