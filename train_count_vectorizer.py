import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import utils.avgLen as avgLen
from utils.punctuations import DataFramePunctuations
from utils.topPunctExtractor import TopPunctExtractor
from utils.topWords import TopWordsCalculator
from joblib import dump, load

# df = pd.read_csv("dataset/fake_or_real_news.csv")
df = pd.read_csv('dataset/DataSet.csv')

def clean_csv(dataframe):
    dataframe['class'] = df['label'].map({'FAKE': 0, 'REAL': 1})
    dataframe['avg_len'] = df['text'].apply(avgLen.getAvgLen)

def get_top_words_list():
    top_words_calculator = TopWordsCalculator(df)
    top_fake_words, top_real_words = top_words_calculator.get_top_words(300)
    top_fake_words_list = top_fake_words.split()
    top_real_words_list = top_real_words.split()

    all_top_words = top_fake_words_list + top_real_words_list
    top_words_returned = list(set(all_top_words))

    return top_words_returned

clean_csv(df)

punctuations_builder = DataFramePunctuations(df)
punctuation_columns = punctuations_builder.add_punctuation_columns()

############################################
# punct_extractor = TopPunctExtractor(df)
# punct_extractor.generateTopPunct()

top_words = get_top_words_list()
dump(top_words, 'app/top_words.joblib')

vectorizer = CountVectorizer(vocabulary=top_words)

df.fillna({'text': ''}, inplace=True)
texts = df['text'].apply(lambda x: ' '.join(sent_tokenize(str(x))) if isinstance(x, str) else '')
x_text = vectorizer.fit_transform(texts)

x = pd.DataFrame(x_text.toarray(), columns=vectorizer.get_feature_names_out())
df_features = df[punctuation_columns + ['avg_len']]

x = pd.concat([x, df_features], axis=1)

titles = df['title'].fillna('')
x_title = vectorizer.fit_transform(titles)
x_title_df = pd.DataFrame(x_title.toarray(), columns=vectorizer.get_feature_names_out())

x = pd.concat([x, x_title_df], axis=1)
y = df['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(n_jobs=-1)
model.fit(x_train, y_train)
print(f"Acuratete train : {100 * model.score(x_train, y_train)}")

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acuratete test : {100 * accuracy}")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

dump(model, 'app/random_forest_model.joblib')