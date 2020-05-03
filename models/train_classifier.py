"""
Disaster Response Piplenine Project

ML Pipeline Preparation

The script loads sql database, splits into training and validation sets. Creates, trains, validate the classification model. Finally, stores model as a pikle file.

Args:
model.pkl
DisasterResponse.db

Run:
python3 train_classifier.py ../data/DisasterResponse.db model.pkl
"""

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
import pandas as pd
import re
import pickle
import numpy as np
from time import time
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import sys


def load_data(database_filepath):
    """Function reads sql database into pandas dataframe and splits into 
    input and output variables, labels.
    Args:
    database_filepath -> sql database

    Returns:
    X -> input variable
    y -> output variables
    category_names -> labekls for classification
    """ 
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df['message']
    y = df[df.columns[4:]]
    
    return X, y, list(y.columns)



def tokenize(text):
    """Function takes a raw text and removes all non-alphabetic and non-numeric
    characters, strips white spaces and  converts to lower case. Preprocessed 
    words returned as a base form.

    Args:
    text -> raw text

    Returns:
    clean_tokens -> tokenized text
    """ 
    # kee[ only letters and numbers
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []

    #clean text and split into tokens(words)
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """Function constracts a pipline for the classification model

    Args:
    None

    Returns:
    cv -> greedsearch pipline for the model
    """ 
    # Define a pipeline combining a text feature extractor with a     #MultiOutputClassifier
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf' , MultiOutputClassifier(RandomForestClassifier()))
        ])

    parameters = {
        #'vect__max_df': (0.5, 0.75),
        #'vect__max_features': (None, 5000, 10000, 50000),
        'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
        #'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        #'clf__estimator__n_estimators': [10]   
        }

    #find the best parameters for both the feature extraction and the
    # classifier
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, 
                  cv=3,n_jobs=-1,verbose=10)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Function perfoems prediction and evaluate the performance of the model.

    Args:
    model -> classification model
    X_test -> imput parameters, validation set
    y_test -> output parameters, validation set
    category_names -> labels
    """ 
    # predict on test data
    Y_pred =model.predict(X_test)

    # calculate and print classification performance for each category
    print(classification_report(Y_test.values,Y_pred,target_names =category_names))
    
    #calculate the precision and accuracy score

    print("Precision score: {}".format(precision_score(Y_test.values,Y_pred,
                                                  average='weighted')))
    print("Accuracy score:{}".format(metrics.accuracy_score(Y_test.values,Y_pred)))


def save_model(model, model_filepath):
    """Function saves our classification model as a pickle file.

    Args:
    model -> classification model
    model_filepath -> path for pickle model

    """ 

    pickle.dump(model, open('model.pkl', 'wb'))


def main():
    """Main function. Initiate execution of the script. 

    Args:
    None
    Returns:
    None
    """ 

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
