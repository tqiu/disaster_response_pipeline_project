import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    # create a dataframe of the 36 individual category columns
    new = df.categories.str.split(';', expand=True)
    # extract a list of new column names for categories
    cat_list = df.categories[0].split(';')
    cat_name = [cat[:-2] for cat in cat_list]
    new.columns = cat_name

    for column in new.columns:
    # set each value to be the last character of the string
        new[column] = new[column].str[-1]
        # convert column from string to numeric
        new[column] = new[column].astype(int)

    # concatenate the original dataframe with the new dataframe
    df = pd.concat([df, new], axis=1)
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)

    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///data/DisasterResponseDatabase.db')
    df.to_sql(database_filename, engine, index=False)


def main():
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
