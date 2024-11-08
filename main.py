import os
from api.app import create_app

if __name__ == "__main__":
    app = create_app(os.environ.get('FLASK_ENV', 'default'))
    app.run(host="0.0.0.0", port=5000, debug=app.config['DEBUG'])
