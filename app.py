import os
from flask import Flask, render_template
from flask_cors import CORS
from api.extensions import cache
from api.routes import api_bp

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev')  # Use environment variable

# Configure CORS
CORS(app)

# Configure caching
cache.init_app(app, config={
    'CACHE_TYPE': 'SimpleCache',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Register blueprints
app.register_blueprint(api_bp, url_prefix='/api')

# Root route for documentation
@app.route('/')
def index():
    return render_template('docs.html')
