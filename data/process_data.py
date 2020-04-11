import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    This function reads messages and categories files and merges them
    
    Input:
    messages_filepath - path to messages csv file
    categories_filepath - path to categories csv file
    
    Output:
    df - a DataFrame object obtained by merging messages and categories
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on = 'id')
    return df

def clean_data(df):
    """
    This function cleans the DataFrame object and returns cleaned version
    Steps performed:
    1. Splits categories variable initially obtained as a raw string into 36 columns, 
       representing each category
    2. Drops duplicate data
    
    Input:
    df - raw data in the form of DataFrame object
    
    Output:
    df - cleaned dataframe
    """
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0, :]
    category_colnames = row.apply(lambda row: row[:-2])
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)

    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Connects to database and exports given dataframe to tweets table
    
    Input:
    df - DataFrame to upload
    database_filename - path to the database
    
    Output:
    none
    """
    engine = create_engine('sqlite:///{0}'.format(database_filename))
    df.to_sql('tweets', engine, index=False)  


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