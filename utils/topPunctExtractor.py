from string import punctuation

class TopPunctExtractor():
    def __init__(self, df):
        self.df = df
        
    def generateTopPunct(self):
        punctuation_set = {p : {"fake" : 0, "real" : 0} for p in punctuation}
        for punct in punctuation:
            for index, row in self.df.iterrows():
                if row['label'] == 'FAKE':
                    punctuation_set[punct]['fake'] += row['text'].count(punct)
                else:
                    punctuation_set[punct]['real'] += row['text'].count(punct)
        with open("./punctuations.txt", "w") as file:
            for punct in punctuation:
                total_punct_count = punctuation_set[punct]['real'] + punctuation_set[punct]['fake']
                file.write(f"Punct : {punct} -- Real({punctuation_set[punct]['real'] / total_punct_count * 100:.1f}%) <-> Fake({punctuation_set[punct]['fake'] / total_punct_count * 100:.1f}%)\n")