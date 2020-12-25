# Load required libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
import dill as pickle
from customized_feature import GetReadabilityIndex
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


n_class = 36

def load_data(database_filepath):
    """ 
    Load data from a database into a dataframe 
    ------
    Parameters:
        database_filepath (str): relative filepath for the database
    ------
    Returns:
        X (numpy 1d array): messages as predictor
        Y (numpy 2d array): classification labels 0, 1 or 2
        category_names (list): category names
    """
    df = pd.read_sql_table(database_filepath, f'sqlite:///{database_filepath}')
    X = df['message'].values
    Y = df.iloc[:,-n_class:].values
    category_names = df.columns[-n_class:]
    return X, Y, category_names


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


def build_model():
    """ Build an NLP & ML pipeline with grid search """

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

    grid_search = GridSearchCV(pipeline, param_grid=params, cv=3)

    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):
    """ 
    Make predictions on the test set and print a classification report for each category
    ------
    Parameters:
        model: trained model
        X_test (numpy 1d array): messages as predictor
        Y_test (numpy 2d array): classification labels 0, 1 or 2
        category_names (list): category names
    ------
    Returns:
        None. Print out each category name and the corresponding classification report with precison, 
        recall and f1-score
    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test[:,i], Y_pred[:,i]))


def save_model(model, model_filepath):
    """ Save model to a pickle file """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """ Execute the entire ML pipeline """
    
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
