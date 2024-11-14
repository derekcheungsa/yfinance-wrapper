import os
from api.app import create_app


def main():
    # Get the environment configuration
    env = os.environ.get('FLASK_ENV', 'default')
    
    # Create the application instance
    app = create_app(env)
    
    # Run the application
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=app.config['DEBUG']
    )

if __name__ == "__main__":
    main()

