import pandas as pd
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("dataset/fake_or_real_news.csv")

df.drop(labels=['id'], axis='columns', inplace=True)
df['class'] = df['label'].map({'FAKE': 0, 'REAL': 1})

###################################################################################################
def getAvgLen(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return 0
    return sum(len(sent) for sent in sentences) / len(sentences)

df['avg_len'] = df['text'].apply(getAvgLen)


################################################# this is for XGB Classifier
# replace_map = {
#     '<': 'lt', '>': 'gt', '[': 'lb', ']': 'rb', '{': 'lc', '}': 'rc',
#     '(': 'lp', ')': 'rp', '&': 'amp', '|': 'pipe', '#': 'hash', '%': 'perc',
#     '*': 'star', '@': 'at', '!': 'excl', '?': 'qmark', ':': 'colon', ';': 'semi'
# }
# for punct in punctuation:
#     safe_punct = replace_map.get(punct, punct)
#     col_name = f'punct_{safe_punct}'
#     df[col_name] = df['text'].apply(lambda text: text.count(punct))
# punctuation_columns = [col for col in df.columns if col.startswith('punct_')]

#################################################
for punct in punctuation:
    col_name = punct.replace(punct, f'punct_{punct}')
    df[col_name] = df['text'].apply(lambda text: sum([str(text).count(punct)]))
punctuation_columns = [col for col in df.columns if col.startswith('punct_')]
###################################################################################################


###################################################################################################
# punctuation_set = {p : {"fake" : 0, "real" : 0} for p in punctuation}
# for punct in punctuation:
#     for index, row in df.iterrows():
#         if row['label'] == 'FAKE':
#             punctuation_set[punct]['fake'] += row['text'].count(punct)
#         else:
#             punctuation_set[punct]['real'] += row['text'].count(punct)
# with open("./punctuations.txt", "w") as file:
#     for punct in punctuation:
#         total_punct_count = punctuation_set[punct]['real'] + punctuation_set[punct]['fake']
#         file.write(f"Punct : {punct} -- Real({punctuation_set[punct]['real'] / total_punct_count * 100:.1f}%) <-> Fake({punctuation_set[punct]['fake'] / total_punct_count * 100:.1f}%)\n")
###################################################################################################


###################################################################################################
stop_words = stopwords.words('english')
fake_text_words = []
real_text_words = []
lemmatizer = WordNetLemmatizer()

for index, row in df.iterrows():
    words = word_tokenize(row['text'].lower())
    lemmatized_words = []
    for word in words:
        if word not in stop_words and word.isalpha() and word not in punctuation:
            lemmatized_words.append(lemmatizer.lemmatize(word))
    if row['label'] == 'FAKE':
        fake_text_words.extend(lemmatized_words)
    elif row['label'] == 'REAL':
        real_text_words.extend(lemmatized_words)

fake_counter = Counter(fake_text_words)
real_counter = Counter(real_text_words)

top_fake_words = ' '.join(word for word, _ in fake_counter.most_common(150))
top_real_words = ' '.join(word for word, _ in real_counter.most_common(150))
top_words = list(set(top_fake_words + top_real_words))

vectorizer = CountVectorizer(vocabulary=top_words)

texts = df['text'].apply(lambda x: ' '.join(sent_tokenize(str(x))) if isinstance(x, str) else '')
x_text = vectorizer.fit_transform(texts)
###################################################################################################


###################################################################################################
x = pd.DataFrame(x_text.toarray(), columns=vectorizer.get_feature_names_out())
x = pd.concat([x, df[punctuation_columns]], axis=1)
x = pd.concat([x, df['avg_len']], axis=1)

y = df['class']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

################################################# scade acuratete cu 1% pe random forest
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

################################################# RANDOM FOREST CLASSIFIER 99% TRAIN 86% TEST
model = RandomForestClassifier()
model.fit(x_train, y_train)
print(f"Acuratete train : {100 * model.score(x_train, y_train)}")

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Acuratete test : {100 * accuracy}")

################################################# XGB CLASSIFIER 90% TRAIN 85% TEST
# model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, eval_metric='logloss')
# model.fit(x_train, y_train)

# y_pred_test = model.predict(x_test)
# y_pred_train = model.predict(x_train)
# accuracy = accuracy_score(y_test, y_pred_test)
# print(f"Acuratete train: {100 * accuracy_score(y_train, y_pred_train)}")
# print(f"Acuratete test: {100 * accuracy}")