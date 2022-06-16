# Twitter Sentiment Analysis 
> This project is Sentiment Analysis based on Twitter comments. The code is written in Python and we have used Jupyter Notebook.

>Here I have used two Python Libary for Sentiment Analysis and Data Visualization

- Scikit Learn (sklearn)
- Tensorflow
- scattertext & spacy

# Scikit Learn
> NOTE: We are using lbfgs solver for Logictic Regrerssion Model. lbfgs stand for: "LimitedMemory-Broyden–Fletcher–Goldfarb–Shanno Algorithm". It is one of the solvers' algorithms provided by Scikit-Learn Library. The term limited-memory simply means it stores only a few vectors that represent the gradients approximation implicitly. It has better convergence on relatively small datasets. So I have used max_iter value as 1000.

We have imported below moduled from sklearn libary.
```py
from sklearn.model_selection import train_test_split # function for splitting data to train and test sets
#feature extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression #Logistic Regression Model
from sklearn.pipeline import Pipeline 
from sklearn.metrics import accuracy_score, precision_score, recall_score
```
# TensorFlow
>Here we have used LSTM Model (Long Short Term Memory)

We have imported below moduled from tensorflow libary
```py
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
```