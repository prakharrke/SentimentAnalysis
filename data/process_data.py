import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(filepath):
    '''
    input:
    messages_filepath: The path of messages dataset.
    categories_filepath: The path of categories dataset.
    output:
    df: The merged dataset
    '''
    df = pd.read_csv(filepath)
    
    return df


def clean_data(df, sample_size):
	
    '''
    input:
    df: The merged dataset in previous step.
    sample_size: Size of the sample needed
    output:
    df: Dataset after cleaning.
    '''
    df.columns = ['rating', 'title', 'text']
    reduced_sample = sample_size//2
    print('reduced sample ' , reduced_sample)
    df_sam =  df[df.rating < 3].sample(n=reduced_sample, random_state=42)

    df_sam = pd.concat([df_sam, df[df.rating > 3].sample(n=reduced_sample, random_state=42)])

    df_sam.columns = ['rating', 'title', 'text']

    df_sam.rating = df_sam.rating.apply(convertToLabel)
    
    return df_sam

def save_data(df, database_filename, table_name):

    '''
    Input: 
    df: dataset
    database_filename: name of the file you want the database file to be created with
    table_name: name of the table in the database for given dataset
    '''
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, index=False)  

def convertToLabel(rating) :
    '''
    Input: rating
    output: label with respect to the rating value
    '''
    if rating > 3 :
        return 'Positive'
    else :
        return 'Negative'


def main():
    if len(sys.argv) == 4:

        train_filepath, test_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    Train: {}\n    Test: {}'
              .format(train_filepath, test_filepath))
        df_train = load_data(train_filepath)
        df_test = load_data(test_filepath)


        print('Cleaning data...')
        df_train_sam = clean_data(df_train, 60000)
        print("train shape ", df_train_sam.shape)
        df_test_sam = clean_data(df_test, 12000)
        print("test shape ", df_test_sam.shape)

        print("train distribution ", df_train_sam.rating.value_counts())
        print("test distribution ", df_test_sam.rating.value_counts())

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))

        save_data(df_train_sam, database_filepath, 'reviews_train')
        save_data(df_test_sam, database_filepath, 'reviews_test')
        
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