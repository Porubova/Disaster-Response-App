import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Function takes a raw text and removes all non-alphabetic and non-numeric
    characters, strips white spaces and  converts to lower case. Preprocessed 
    words returned as a base form.

    Args:
    text -> raw text

    Returns:
    clean_tokens -> tokenized text
    """ 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/InsertDatabaseName.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/model.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
   
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    related= df.loc[df.related_category==1].groupby('genre').sum()[df.columns[4:]]
    total = related.sum().sort_values(ascending=False).to_frame().T
    related_sorted = pd.concat([total, related]).iloc[1:, 1:11]
    labels = list(related_sorted.columns)
    
    # create visuals
   
    graphs = [
    
                 {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                ), 
                        
                   
                        
                        
                        
            ],

            'layout': {
                    
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                    
            }
                
                }
        },
                    {
            'data': [
                Bar(
                    y=labels,
                    x=related_sorted.iloc[0], 
                    orientation = 'h',
                    name='Direct'
                ),
                        Bar(
                    y=labels,
                    x=related_sorted.iloc[1], 
                    orientation = 'h',
                    name='News'
                    
                    
                   
                ),
                                    Bar(
                    y=labels,
                    x=related_sorted.iloc[2], 
                    orientation = 'h',
                    name='Social'
                    
                    
                   
                ),
            ],

            'layout': {
                'title': 'Distribution of Relevant Messages by Genre, Top 10',
                'yaxis': {
                    'title': "Category"
                },
                'xaxis': {
                    'title': "Count"
                },
                'margin':{
                        'l':200
            },
                        'height':700,
                        'barmode':'stack'
            }
        }
                
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()