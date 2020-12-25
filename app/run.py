# Load required libraries
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from flask import Markup
import plotly.graph_objs as gos
from plotly.offline import plot
import joblib
from sqlalchemy import create_engine
from customized_feature import GetReadabilityIndex


app = Flask(__name__)

def tokenize(text):
    """ Returns a list of tokens for the input text (str) """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponseDatabase.db')
df = pd.read_sql_table('data/DisasterResponseDatabase.db', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ Render web pages with plotly graphs which provides an overview of data """

    ######################### Visual 1 ################################
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    my_plot_div = plot({"data": [gos.Bar(x=genre_names, y=genre_counts)],
                        "layout": gos.Layout(title={"text": "Distribution of Message Genres", 'x': 0.5},
                        xaxis_title="Genre", yaxis_title='Count')}, 
                        output_type='div')


    ######################### Visual 2 ################################
    # make a dataframe to prepare for the following stacked bar plot
    class_num = [0, 1]
    class_label = df.columns[-36:].tolist()
    class_dict = {}
    for label in class_label:
        class_count = []
        for num in class_num:
            class_count.append((df[label]==num).sum())
        class_dict[label] = class_count
    df_class = pd.DataFrame(class_dict, index=class_num)

    # make a stacked bar plot for all 36 message classes
    graphs = []
    graphs.append(
        {
            'data': [
                gos.Bar(
                    x=class_label,
                    y=df_class.loc[0],
                    name=class_num[0]
                ),
                gos.Bar(
                    x=class_label,
                    y=df_class.loc[1],
                    name=class_num[1]
                )
            ],

            'layout': {
                'title': 'Distribution of 36 Message Classes',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'tickangle': 45
                },
                'barmode': 'stack',
                'height': 500,
                'margin': dict(l=50, r=50, t=60, b=150)
            }
        }
    )


    ######################### Visual 3 ################################
    # calculate the cross-tab dataframe
    df_tab = pd.crosstab(df['related'], df['aid_related'], normalize=True)
    z1 = (df_tab.loc[0] * 100).round(2).tolist()
    z2 = (df_tab.loc[1] * 100).round(2).tolist()

    # make a heatmap showing the cross-tabulation between 'related' and 'aid_related' messages
    graphs.append(
        {
            'data': [
                gos.Heatmap(
                    x=['0', '1'],
                    y=['0', '1'],
                    z=[z1, z2],
                    colorscale='Blues',
                    hoverinfo="z",
                    hovertemplate='%{z}%<extra></extra>',
                    colorbar={'ticksuffix': '%'}
                )
            ],
            'layout': {
                'title': "Cross Tabulation of Related and Aid-related Messages",
                'yaxis': {
                    'title': "Related",
                    'type': 'category'
                },
                'xaxis': {
                    'title': "Aid-related",
                    'type': 'category'
                }
            }
        }
    )

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, my_plot_div=Markup(my_plot_div), graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """ Render web page that handles user query and displays model results"""
    
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()
