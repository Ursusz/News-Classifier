from flask import Flask
from app.controllers.app import main_bp
import os
from dotenv import load_dotenv
from app.extensions import db

def create_app():
    app = Flask(__name__)
    load_dotenv()

    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

    db.init_app(app)

    from app.controllers.app import main_bp
    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()

    return app