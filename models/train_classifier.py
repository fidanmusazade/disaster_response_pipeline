import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import sys
import pandas as pd
import numpy as np
import re
import pickle
import joblib

from sqlalchemy import create_engine

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

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


def load_data(database_filepath):
    """
    This file loads data from sqlite database and separates it into X and Y
    
    Input:
    database_filepath - path to the database containing tweets table
    
    Output:
    X - message column, which will be the input for the model
    Y - the dependent variable (category labels)
    category_names - names of the categories
    """
    engine = create_engine('sqlite:///{0}'.format(database_filepath))
    df = pd.read_sql_table('tweets', con=engine)
    to_drop = ['id', 'message', 'original', 'genre']
    X = df['message']
    Y = df.drop(to_drop, axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names


def tokenize(text):
    """
    This function tokenizes text by deleting punctuation, converting
    to lowercase, deleting stopwords, lemmatizing and stemming
    
    Input:
    text - text to be tokenized
    
    Output:
    stemmed - the resulting tokens
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    
    lemmatized = [lemmatizer.lemmatize(word) for word in words]
    stemmed = [stemmer.stem(w) for w in lemmatized]
    return stemmed


def build_model():
    """
    This function creates a pipeline, defines set of parameters and creates a
    GridSearchCV
    
    Input:
    none
    
    Output:
    cv - GridSearchCV object (model)
    """
    pipeline = Pipeline([
       ('features', FeatureUnion([
           ('nlp', Pipeline([
               ('vect', CountVectorizer(tokenizer=tokenize)),
               ('tfidf', TfidfTransformer())
           ])),
           ('verbcount', VerbCounter())
       ])),
       ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])

    
    parameters = {
#         'features__nlp__vect__ngram_range': [(1,2), (2,2)],
       'features__nlp__tfidf__smooth_idf': [True, False],
       'clf__estimator__n_estimators': [100, 200],
       'clf__estimator__criterion': ['entropy']
    }

    cv = GridSearchCV(pipeline, parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function prints classification report for each of the
    categories
    
    Input:
    model - model to evaluate
    X_test - test features
    Y_test - test labels
    category_names - names of categories in label
    
    Output:
    none
    """ 
    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    for col in category_names:
        print(col)
        print(classification_report(Y_test[col], Y_pred[col]))


def save_model(model, model_filepath):
    """
    This function exports the trained model into a pickle file
    
    Input:
    model - the trained model to be exported
    model_filepath - the path to which to export the model
    """
    joblib.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
