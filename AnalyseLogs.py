import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request

# Create a sample dataframe with log data
data = {'timestamp': ['01-01-2022 01:00:00', '01-01-2022 02:00:00', '01-01-2022 03:00:00','01-01-2022 04:00:00'],
        'level': ['error','warning','info','debug'],
        'message': ['server crashed','network issue','database connection lost','high cpu usage']}
df = pd.DataFrame(data)

# Preprocess the log data
df = df[df.columns.intersection(['timestamp', 'level', 'message'])]

# Create features
vectorizer = CountVectorizer(min_df=2)
X = vectorizer.fit_transform(df['message'])

# Split the data into training and testing sets
X_train, X_test, y
