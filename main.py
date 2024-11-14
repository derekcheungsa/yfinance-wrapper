import os
from api.app import create_app

app = create_app(os.environ.get('FLASK_ENV', 'default'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
