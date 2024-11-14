import os

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY', 'dev')
    CACHE_TYPE = 'SimpleCache'
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_THRESHOLD = 1000  # Maximum number of items the cache will store
    CACHE_KEY_PREFIX = 'yf_'  # Prefix for all cache keys
    JSON_SORT_KEYS = False
    CORS_HEADERS = 'Content-Type'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    CACHE_TYPE = 'RedisCache'
    CACHE_REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
    CACHE_REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
    CACHE_REDIS_DB = 0
    CACHE_OPTIONS = {
        'socket_timeout': 3,
        'socket_connect_timeout': 3,
        'retry_on_timeout': True,
        'max_connections': 20
    }

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    TESTING = False
    CACHE_NO_NULL_WARNING = True

class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    CACHE_TYPE = 'NullCache'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
