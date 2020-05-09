import pandas as pd
import numpy as np
import nltk
import os
import sys, pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sqlalchemy import create_engine






def load_data(database_filepath):
    '''
    Fucntion to load the database from the given filepath and process them as X, y and category_names
    Input: Databased filepath
    Output: Returns the Features X & target y along with target columns names catgeory_names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df_train = pd.read_sql_table('reviews_train', engine)
    df_test = pd.read_sql_table('reviews_test', engine)
    
    X_train = df_train['text']
    y_train = df_train['rating']

    X_test = df_test['text']
    y_test = df_test['rating']

   
    return X_train, X_test, y_train, y_test

def tokenize(text):
    '''
    Function to tokenize the text messages
    Input: text
    output: cleaned tokenized text as a list object
    '''
    # Convert the text to lower case
    text = text.lower()
    #Remove punctuations
    text_normalized = re.sub(r"[^a-zA-Z0-9]", " ", text)
    #Tokenizing the normalized text
    tokens = word_tokenize(text_normalized)
    #removal of stop words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stopwords.words('english')]
    return clean_tokens   





def build_model():
    '''
    Function to build a model, create pipeline, hypertuning as well as gridsearchcv
    Input: N/A
    Output: Returns the model
    '''
    pipeline = Pipeline([
    
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1, 2))),
                ('tfidf', TfidfTransformer())
            ]))
        
        ])),
    
        ('clf', RandomForestClassifier(n_estimators=100, n_jobs=-1))

    ])

    return pipeline


def evaluate_model(model, X_test, y_test):
    '''
    Function to evaluate a model and return the classificatio and accurancy score.
    Inputs: Model, X_test, y_test, Catgegory_names
    Outputs: Prints the Classification report & Accuracy Score
    '''
    y_pred = model.predict(X_test)
    #print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score 
    print('Accuracy Score: {}'.format(np.mean(y_test.values == y_pred)))   

    print("**** Following are F1, precision and recall for the obtained predictions and the test set ****") 
    compute_report(y_test, y_pred)

def compute_report(y_test, y_pred) : 
    '''
    Function to output f1 score, precision and recall for each category in the test set
    Input: test set and predicted result of test set given by model
    Output: No output
    '''
    res = classification_report(y_test, y_pred)
    print(res)



def save_model(model, model_filepath):
    '''
    Function to save the model
    Input: model and the file path to save the model
    Output: save the model as pickle file in the give filepath 
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X_train, X_test, y_train, y_test = load_data(database_filepath)

        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        start_time = time.time()
        model.fit(X_train, y_train)
        print('time taken to train model : {}'.format(time.time() - start_time))
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test)

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