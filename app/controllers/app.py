from flask import request, render_template, Blueprint
from utils.classifiers import predict_rfc, predict_bert

main_bp = Blueprint('main', __name__)

@main_bp.route('/', methods=['GET', 'POST'])
@main_bp.route('/home', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        text = request.form.get('input_text')
        title = request.form.get('input_title')
        if request.form.get('model') == 'rfc':
            prediction = predict_rfc.predict_rfc(text, title)
            return render_template('index.html', prediction_result=prediction, used_model="Random Forest Classifier")
        elif request.form.get('model') == 'brt':
            prediction = predict_bert.predict_bert(text, title)
            return render_template('index.html', prediction_result=prediction, used_model="BERT")

    return render_template('index.html')