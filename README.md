# Twitter Sentiment Analysis 
> This project is Sentiment Analysis based on Twitter comments. The code is written in Python and we have used Jupyter Notebook.

>Here I have used two Python Libary for Sentiment Analysis

- Scikit Learn (sklearn)
- Tensorflow

# Scikit Learn
We have imported below moduled from sklearn libary
```py
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets
#feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression #Logistic Regression Model
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, precision_score, recall_score
```
# TensorFlow
We have imported below moduled from tensorflow libary
