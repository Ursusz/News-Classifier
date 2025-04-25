from flask import request, render_template, Blueprint
from utils.classifiers import predict_rfc, predict_bert
from app.models.models import DomainName, News
import validators
from urllib.parse import urlparse

main_bp = Blueprint('main', __name__)

def verify_url_in_bd(url):
    if validators.url(url):
        parsed_url = urlparse(url)
        domain_name = parsed_url.netloc
        path = parsed_url.path

        domain_entry = DomainName.query.filter_by(domain_name=domain_name).first()
        if domain_entry:
            path_entry = News.query.filter_by(news=path).first()
            if path_entry:
                prediction = path_entry.label
                return prediction
            else:
                return None
        else:
            return None
    return None

@main_bp.route('/', methods=['GET', 'POST'])
@main_bp.route('/home', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@main_bp.route('/predict', methods=['POST'])
def predict():
    prediction = None
    if request.method == "POST":
        input_type = request.form.get('input_type')
        model = request.form.get('model')
        used_model = 'Random Forest Classifier' if model == 'rfc' else 'BERT'

        if input_type == 'url':
            url = request.form.get('input_url')
            prediction = verify_url_in_bd(url)
            if prediction is not None:
                print('Here is not none')
                return render_template('index.html', prediction_result=prediction, used_model='DataBase')
            else:
                print('Here is none')
                # TODO: get the text and title from url
                return render_template('index.html', prediction_result='no pred', used_model=used_model)

        elif input_type == 'txt':
            text = request.form.get('input_text')
            title = request.form.get('input_title')
            print(model)
            if model == 'rfc':
                prediction = predict_rfc.predict_rfc(text, title)
                print(prediction)
            elif model == 'brt':
                prediction = predict_bert.predict_bert(text, title)

        return render_template('index.html', prediction_result=prediction, used_model=used_model)

    return render_template('index.html')