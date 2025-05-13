from flask import request, render_template, Blueprint, redirect, url_for, flash, session
from flask_login import login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from utils.classifiers import predict_rfc, predict_bert
from app.models.user import USERS
from app.extensions import db

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    session.pop('_flashes', None)
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        user = USERS.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash("Login successful!", "success")
            return redirect(url_for("main.index"))
        flash("Login failed. Check credentials.", "danger")
    return render_template("login.html")


@auth_bp.route("/logout")
def logout():
    session.pop('_flashes', None)
    logout_user()
    return redirect(url_for("main.index"))


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        confirm_password = request.form["confirm_password"]

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("auth.register"))

        existing_user = USERS.query.filter_by(username=username).first()
        if existing_user:
            flash("Username already taken. Please choose another one.", "danger")
            return redirect(url_for("auth.register"))

        hashed_password = generate_password_hash(password)
        user = USERS(username=username, email=email, password_hash=hashed_password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("auth.login"))
    return render_template("register.html")