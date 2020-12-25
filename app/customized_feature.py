# Add a Readability index (Gunning-Fog Grade Index) feature

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def tokenize(text):
    """ Returns a list of tokens for the input text (str) """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


class GetReadabilityIndex(BaseEstimator, TransformerMixin):
    """
    A class to create a Gunning-Fog Grade Index (readability index)
    ------
    Methods:
        sentence_count(text): count number of sentences in the text
        word_count(text): count number of tokens in the text
        hard_word_count(text): count number of hard words in the text
        fog_index(text): compute the Gunning-Fog Grade Index for the text
        fit(x, y=None): return the same fit function from BaseEstimator
        transform(X): apply the fog_index(text) to X, transform X and return the fog_index 
            in a dataframe
    """

    def sentence_count(self, text):
        sent = sent_tokenize(text)
        return len(sent)

    def word_count(self, text):
        word = tokenize(text)
        return len(word)

    def hard_word_count(self, text):
        words = tokenize(text)
        wnl = WordNetLemmatizer()
        words = [wnl.lemmatize(w, pos='v') for w in words]
        hard_word = [w for w in words if len(w) >= 3]
        return len(hard_word)

    def fog_index(self, text):
        n_sent = self.sentence_count(text)
        n_word = self.word_count(text)
        if n_sent == 0 or n_word == 0:
            return 0
        n_hard_word = self.hard_word_count(text)
        index = 0.4 * (n_word/n_sent + 100 * (n_hard_word/n_word))
        return index

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_index = pd.Series(X).apply(self.fog_index)
        return pd.DataFrame(X_index)
