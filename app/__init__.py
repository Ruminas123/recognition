# app/__init__.py

from flask import Flask
from .config import Config
from .extensions import init_mongo

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config['DEBUG'] = True
    init_mongo(app)
    from .routes.routes import main_routes
    app.register_blueprint(main_routes)
    return app
