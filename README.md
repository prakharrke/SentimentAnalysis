# SentimentAnalysis


## Problem statement

Build a sentiment analysis model that classifies the input text as either Positive or Negative. The input would be reviews or feedbacks of users but can very well be extended to other kinds of text messages like social media posts or tweets. This is intended to be achieved by training our model on a dataset of already labeled Amazon product reviews.

## For a detailed explanation, please read the Medium article written by me, which documents all aspects of this project.  https://medium.com/@dixitprakhar94/sentiment-analysis-building-from-the-ground-up-e12e9195fac4


## Project Structure 

Project contains following folders
1. data - data folder contains process_data.py file which is responsible for loading, cleaning and storing the data in db file.

2. models - models folder contains train_classifier.py file which builds the model, trains it on the dataset saved in the db and creates a pickle file out of it.

3. app - app folder contains run.py file which is responsible for running the web app and predicting user inputs. It 

4. app/templates - templates folder contains the html files for the web app.


## Steps to run the project

### The dataset was too big to put in the GitHub repositiry. Hence I have uploaded it on my google drive and following is the link to it. Please download, unzip and paste train.csv and test.csv files inside data folder.

https://drive.google.com/drive/folders/1vMRBSxm5JuvAcAPrwr1W0hFN_EgxiKSI?usp=sharing

Once you put the dataset files in data folder, run the following commands:

### 1. `python data/process_data.py data/train.csv data/test.csv data/AmazonReviews.db` 

This command will load data from train.csv and test.csv, clean it and create AmazonReviews.db file in data folder.

### 2. `python models/train_classifier.py data/AmazonReviews.db models/classifier.pkl`

This command will create a pickle file of the model and save it in models folder.

Once both the steps have been successfully run, it is time to start the web app.

### Go into app folder using `cd app` and then run the command `python run.py` 

This will start the web app. You can access the webapp at 
## `http://localhost:3001/`
