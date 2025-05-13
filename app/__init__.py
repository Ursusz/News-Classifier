from flask import Flask
from app.controllers.app import main_bp
import os
from dotenv import load_dotenv
from app.extensions import db
from flask_login import LoginManager

def create_app():
    app = Flask(__name__)
    load_dotenv()

    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')

    db.init_app(app)

    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    from app.models.user import USERS as User

    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))

    from app.controllers.app import main_bp
    app.register_blueprint(main_bp)

    from app.controllers.auth import auth_bp
    app.register_blueprint(auth_bp)

    # with app.app_context():
    #     db.create_all()

    return app