import os
from app import app

if __name__ == "__main__":
    app.config['DEBUG'] = os.environ.get('FLASK_ENV', 'default') == 'development'
    app.run(host="0.0.0.0", port=5000, debug=app.config['DEBUG'])
