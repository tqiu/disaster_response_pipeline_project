# Disaster Response Pipeline Project

## Summary
In this project, a Machine Learning pipeline is built to classify disaster messages. The dataset is from Figure Eight, containing real messages that were sent during disaster events. By categorizing these events, messages can be sent to an appropriate disaster relief agency. In the web app, visualizations of the dataset are displayed. An emergency worker or an organization can input a message and get classification results in several categories (out of 36 categories).

## Instructions
1. Run the following commands in the project's root directory to set up database and model.

    - To run ETL pipeline that cleans data and stores in database \
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseDatabase.db`
    - To run ML pipeline that trains classifier and saves \
        `python models/train_classifier.py data/DisasterResponseDatabase.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app. \
    `python run.py`

3. Go to http://127.0.0.1:5000/

## Files in the Repository
* app \
| - template \
| | - master.html # main page of web app \
| | - go.html # classification result page of web app \
| - run.py # Flask file that runs app \
| - customized_feature.py # a python module containing customized features for the model 
* data \
| - disaster_categories.csv # data to process \
| - disaster_messages.csv # data to process \
| - process_data.py \
| - DisasterResponseDatabase.db # database to save clean data to 
* models \
| - train_classifier.py \
| - classifier.pkl # saved model \
| - customized_feature.py # a python module containing customized features for the model 
* README.md
