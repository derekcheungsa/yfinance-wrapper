import os
from flask import Flask, render_template, jsonify
from flask_cors import CORS
from api.extensions import cache
from api.routes import api_bp
from api.config import config

def create_app(config_name=None):
    """Application factory function."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')

    # Create Flask app instance
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object(config[config_name])

    # Initialize extensions
    CORS(app)
    cache.init_app(app)

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')

    # Register error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({'error': 'Not Found'}), 404

    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal Server Error'}), 500

    @app.errorhandler(429)
    def ratelimit_error(error):
        return jsonify({'error': 'Rate limit exceeded'}), 429

    # Root route for documentation
    @app.route('/')
    @cache.cached(timeout=3600)  # Cache documentation for 1 hour
    def index():
        return render_template('docs.html')

    return app
