"""
Disaster Response Piplenine Project

ETL Pipeline Preparation

The script takes the file paths of the two datasets and database, cleans the datasets, and stores the clean data into a SQLite database in the specified database file path.

Args:
messages.csv
categories.csv 
DisasterResponse.db

Run:
python3 process_data.py messages.csv categories.csv DisasterResponse.db
"""

# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to load data sets an to merge them into one data frame.

    Args:
    messages_filepath -> path to messages csv file
    categories_filepath -> paht to categories csv file

    Returns:
    df -> merged pandas dataframe

    """
    # load messages and categories datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df =  pd.merge(messages, categories, on='id')

    return df 

def change_value(row):
    """Function takes a row value from column and converts values to binary

    Args:
    row-> value from column

    Returns:
    row -> value as 1 or 0

    """
    if row in [1,2]:
        return 1
    else:
        return 0

def clean_data(df):
    """Function splits the categories column into separate, clearly named 
    columns, converts values to binary, and drops duplicates.

    Args:
    df-> pandas dataframe

    Returns:
    df -> cleaned pandas dataframe

    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',  expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = list(row.apply(lambda x: x.split("-")[0])+'_category')

    # rename the columns of `categories`
    categories.columns = category_colnames

    #Convert category to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)

    # drop duplicates
    df.drop_duplicates(keep=False,inplace=True)
    
    # change values in realted_category colum to 1 and 0
    df.related_category=df.related_category.apply(change_value)

    return df


def save_data(df, database_filename):
    """Function splits the categories column into separate, clearly named 
    columns, converts values to binary, and drops duplicates.

    Args:
    df-> pandas dataframe
    database_filename -> destination path to sql database ('InsertDatabaseName.db')
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False)


def main():
    """Main function. Initiate execution of the script. 

    Args:
    None
    Returns:
    None
    """ 

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
