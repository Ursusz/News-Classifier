from flask import request, render_template, Blueprint
from flask_login import login_required, current_user
from utils.classifiers import predict_rfc, predict_bert
from app.models.models import DomainName, News
import validators
from urllib.parse import urlparse
from utils.htmlParser import get_html
from app.extensions import db

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

def add_url_to_db(url, prediction):
    if validators.url(url):
        parsed_url = urlparse(url)
        domain_name = parsed_url.netloc
        path = parsed_url.path

        domain_entry = DomainName.query.filter_by(domain_name=domain_name).first()
        news_entry = News.query.filter_by(news=path).first()
        if not domain_entry:
            domain_entry = DomainName(domain_name=domain_name, real_news=0, fake_news=0)
            db.session.add(domain_entry)
        elif not news_entry:
            news_entry = News(news=path, domain_name=domain_name, label=prediction, user=current_user.username)
            db.session.add(news_entry)
        else:
            if prediction == 1:
                domain_entry.real_news += 1
            elif prediction == 0:
                domain_entry.fake_news += 1

        # news_entry = News(news=path, domain_name=domain_name, label=prediction, user=current_user.username)
        # news_entry = News(news=path, domain_name=domain_name, label=prediction, user=current_user.username)
        # db.session.add(news_entry)
        db.session.commit()
        print(f"URL '{url}' added in database with prediction: {prediction}")
    else:
        print(f"Invalid URL: {url}")


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
            if prediction == 0:
                prediction = 'FAKE'
            elif prediction == 1:
                prediction = 'REAL'
            if prediction is not None:
                return render_template('index.html', prediction_result=prediction, used_model='DataBase')
            else:
                text, title = get_html(url)
                print('here')

                prediction = ''
                if model == 'rfc':
                    prediction = predict_rfc.predict_rfc(text, title)
                    # print(f"Model rfc: {prediction}")
                elif model == 'brt':
                    prediction = predict_bert.predict_bert(text, title)
                    # print(f"Model bert: {prediction}")
                add_url_to_db(url, 0 if prediction == 'FAKE' else 1)
                # print(prediction)
                return render_template('index.html', prediction_result=prediction, used_model=used_model)

        elif input_type == 'txt':
            text = request.form.get('input_text')
            title = request.form.get('input_title')
            if model == 'rfc':
                prediction = predict_rfc.predict_rfc(text, title)
            elif model == 'brt':
                prediction = predict_bert.predict_bert(text, title)

        return render_template('index.html', prediction_result=prediction, used_model=used_model)

    return render_template('index.html')

@main_bp.route("/<username>")
@login_required
def profile(username):
    user_news = db.session.query(News.news, DomainName.domain_name, News.label). \
        join(DomainName, News.domain_name == DomainName.domain_name). \
        filter(News.user == username).all()

    results = [f"{domain}{news} (Label: {"FAKE" if label == 0 else "REAL"})" for news, domain, label in user_news]
    return render_template("profile.html", user=current_user, user_news=results)
