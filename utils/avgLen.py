from nltk.tokenize import sent_tokenize

def getAvgLen(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return 0
    return sum(len(sent) for sent in sentences) / len(sentences)