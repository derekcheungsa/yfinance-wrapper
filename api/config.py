import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'dev')
    CACHE_TYPE = 'SimpleCache'  # Using SimpleCache as it's perfect for single worker setup
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_THRESHOLD = 1000  # Maximum number of items the cache will store
    JSON_SORT_KEYS = False
    CORS_HEADERS = 'Content-Type'
    DEBUG = False
    TESTING = False

class ProductionConfig(Config):
    """Production configuration."""
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 600  # 10 minutes in production

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 300  # 5 minutes in development

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    CACHE_TYPE = 'NullCache'  # Disable caching in testing

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
