from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from train_classifier import tokenize

class VerbCounter(BaseEstimator, TransformerMixin):

    def count_verb(self, text):
        i = 0
        pos_tags = nltk.pos_tag(tokenize(text))
        for (word, tag) in pos_tags:
            if tag in ['VB', 'VBP']:
                i = i + 1
        return i

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.count_verb)
        return pd.DataFrame(X_tagged)