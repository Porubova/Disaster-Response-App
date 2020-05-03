# Disaster Response Application

The project for Udacity's Dtata Scientist Nanodegree Program. 

# Overview
This project contains data sets with real messages that were sent during disaster events. The goal of this project is to create a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

This project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app displays visualizations of the data. 

### Directory Tree and contents
```sh
.
├── app
│   ├── run.py
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── categories.csv
│   ├── messages.csv
│   └── process_data.py
├── models
│   └── train_classifier.py
└── README.md
```
### Project Components

1. ETL Pipeline
The Python script, process_data.py:

  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database

2.  ML Pipeline
The Python script, train_classifier.py:

  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file
3.  Flask Web App

  - Loads DisasterResponse.db
  - Loads Model
  - perform classification of input messages
  - generates two visualizations 


### Installation and Run



Clone git repository
```sh
git clone https://github.com/Porubova/Disaster-Response-App.git
```

Navigate to clonned project workspace
```sh
cd Disaster-Response-App/

```
Create DisasterResponse SQL database
```sh
cd data/
python3 process_data.py messages.csv categories.csv DisasterResponse.db
cd..
```

Build and train model
```sh
cd models/
python3 train_classifier.py ../data/DisasterResponse.db model.pkl
cd..
```
Run web application
```sh
cd app
python3 run.py
```
Go to http://0.0.0.0:3001/

