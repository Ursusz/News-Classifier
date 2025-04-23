import joblib
from nltk import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from utils import avgLen
from utils.punctuations import DataFramePunctuations

model = joblib.load('app/random_forest_model.joblib')
top_words = joblib.load('app/top_words.joblib')

vectorizer = CountVectorizer(vocabulary=top_words)

def predict(text, title):
    try:
        df = pd.DataFrame([{'text': text}])
        punctuations_builder = DataFramePunctuations(df)
        punctuation_columns = punctuations_builder.get_punctuation_columns()

        df['avg_len'] = df['text'].apply(avgLen.getAvgLen)

        texts = ' '.join(sent_tokenize(str(text)))
        x_text = vectorizer.transform([texts])
        x_text_df = pd.DataFrame(x_text.toarray(), columns=vectorizer.get_feature_names_out()) # Fără prefix 'text_'

        titles = ' '.join(sent_tokenize(str(title)))
        x_title = vectorizer.transform([titles])
        x_title_df = pd.DataFrame(x_title.toarray(), columns=vectorizer.get_feature_names_out()) # Fără prefix 'title_'

        df_features = df[punctuation_columns + ['avg_len']]

        x = pd.concat([x_text_df, df_features, x_title_df], axis=1)

        y = model.predict(x)
        result = 'FAKE' if y == 0 else 'REAL'
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None