import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import dill as pickle
from customized_feature import GetReadabilityIndex

n_class = 36

def load_data(database_filepath):
    df = pd.read_sql_table(database_filepath, 'sqlite:///../data/DisasterResponseDatabase.db')
    X = df['message'].values
    Y = df.iloc[:,-n_class:].values
    category_names = df.columns[-n_class:]
    return X, Y, category_names


import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def tokenize(text):
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import RandomizedSearchCV

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('readability_index', GetReadabilityIndex())
        ])),

        ('clf', MultiOutputClassifier(XGBClassifier(random_state=42, n_jobs=-1)))
    ])

    # A parameter grid for XGBoost
    params = {'clf__estimator__n_estimators': [100, 200, 300],
              'clf__estimator__min_child_weight': [1, 5, 10],
              'clf__estimator__gamma': [0, 0.5, 1, 2, 5],
              'clf__estimator__subsample': [0.6, 0.8],
              'clf__estimator__colsample_bytree': [0.6, 0.8],
              'clf__estimator__max_depth': [3, 4, 5]}


    random_search = RandomizedSearchCV(pipeline, param_distributions=params,
                    n_iter=20, random_state=42, cv=3)

    return random_search


from sklearn.metrics import classification_report

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

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
