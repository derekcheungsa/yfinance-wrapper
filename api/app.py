import os
from flask import Flask, render_template
from flask_cors import CORS
from api.extensions import cache
from api.routes import api_bp
from api.config import config

def create_app(config_name=None):
    """Application factory function."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')

    app = Flask(__name__)
    app.config.from_object(config[config_name])

    # Initialize extensions
    CORS(app)
    cache.init_app(app)

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')

    # Root route for documentation
    @app.route('/')
    def index():
        return render_template('docs.html')

    return app
