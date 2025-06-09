import pandas as pd
from string import punctuation

class DataFramePunctuations:
    def __init__(self, df):
        self.df = df

    def add_punctuation_columns(self):
        for punct in punctuation:
            col_name = punct.replace(punct, f'punct_{punct}')
            self.df[col_name] = self.df['text'].apply(lambda text: sum([str(text).count(punct)]))
        punctuation_columns = [col for col in self.df.columns if col.startswith('punct_')]
        return punctuation_columns

    def get_punctuation_columns(self):
        punctuation_columns = self.add_punctuation_columns()
        return punctuation_columns

