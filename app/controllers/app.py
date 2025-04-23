import os
from flask import Flask, request, jsonify, render_template, Blueprint
import joblib
from utils import classiffier

main_bp = Blueprint('main', __name__)

model = joblib.load('app/random_forest_model.joblib')

@main_bp.route('/', methods=['GET', 'POST'])
@main_bp.route('/home', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        text = request.form.get('input_text')
        title = request.form.get('input_title')
        print(f"Am primit : {text}")
        print(classiffier.predict(text, title))

    return render_template('index.html')