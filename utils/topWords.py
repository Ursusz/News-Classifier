from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from string import punctuation
import pandas as pd

class TopWordsCalculator():
    def __init__(self, df):
        self.df = df

    def get_top_words(self, number):
        stop_words = stopwords.words('english')
        fake_text_words = []
        real_text_words = []
        lemmatizer = WordNetLemmatizer()

        for _, row in self.df.iterrows():
            words = word_tokenize(row['text'].lower())
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words and word not in punctuation]
            if row['label'] == 'FAKE':
                fake_text_words.extend(lemmatized_words)
            elif row['label'] == 'REAL':
                real_text_words.extend(lemmatized_words)

        fake_counter = Counter(fake_text_words)
        real_counter = Counter(real_text_words)

        top_fake_words_with_counts = fake_counter.most_common(number)
        top_real_words_with_counts = real_counter.most_common(number)

        top_fake_words = [word for word, _ in top_fake_words_with_counts]
        top_real_words = [word for word, _ in top_real_words_with_counts]

        all_top_words = top_fake_words + top_real_words

        unique_top_words = sorted(list(set(all_top_words)))

        top_fake_unique = [word for word in top_fake_words if word in unique_top_words][:number]
        top_real_unique = [word for word in top_real_words if word in unique_top_words][:number]

        top_fake_words_string = ' '.join(top_fake_unique)
        top_real_words_string = ' '.join(top_real_unique)

        with open("./words.txt", "w") as file:
            file.write(top_real_words_string)
            file.write("\n")
            file.write(top_fake_words_string)

        return top_fake_words_string, top_real_words_string

df = pd.read_csv('D:\\Facultate\\Anul 2\\Semestrul II\\MDS Project\\dataset\\DataSet.csv')

calc = TopWordsCalculator(df)
calc.get_top_words(300)