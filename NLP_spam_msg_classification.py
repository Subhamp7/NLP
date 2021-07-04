#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 23:46:40 2021

@author: subham
"""

#importing reuired libraries
import re
import seaborn as sns
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import metrics

#reading the data and removing useleas data
spam_ham_dataset = pd.read_csv("spam_ham_dataset.csv")
print("The columns are :", spam_ham_dataset.columns)

#dropping the unneccessary columns
spam_ham_dataset =spam_ham_dataset.drop(['Unnamed: 0', 'label'], axis = 1)

#check for empty fields
print("The number of empty fields are ", spam_ham_dataset.isnull().sum() )

#getting the number of spam and ham distribution
print("SPAM and HAM distibution:\n",  spam_ham_dataset.groupby('label_num').count())

#creating a column which consist of the len of text msg
spam_ham_dataset['length']=spam_ham_dataset['text'].apply(len)

#plotting the test type and its length
sns.countplot(x='length',hue='label_num', data=spam_ham_dataset)

#stemming and reoving stop words
lemmatizer = WordNetLemmatizer()

def preprocess_data(input_string):
    review = re.sub('[^a-zA-Z]', ' ',input_string)
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
    return ' '.join(review)

corpus = []
count = 1
for index in range(0, len(spam_ham_dataset)):
    print("Processing {} of {}".format(index, len(spam_ham_dataset)))
    processed_text = preprocess_data(spam_ham_dataset['text'][index])
    corpus.append(processed_text)
    count+=1


#applying tfidvectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,5))
X_dataset = tfidf_v.fit_transform(corpus).toarray()
Y_dataset = spam_ham_dataset.iloc[:,1]


X_train, X_test, Y_train, Y_test = train_test_split(X_dataset, Y_dataset, test_size= 0.25, random_state=0)


#hyperparameter tunning
previous_score=0
alpha_value=[]
for alpha in np.arange(0,1,0.01):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(X_train,Y_train)
    y_pred=sub_classifier.predict(X_test)
    score = metrics.accuracy_score(Y_test, y_pred)
    if score>previous_score:
        classifier=sub_classifier
    alpha_value.append({alpha:score})

#applying naive bayes classifier
nb=MultinomialNB(alpha=0.01)
nb.fit(X_train,Y_train)

#predicting the test data
pred=nb.predict(X_test)

#calculating the metrics
report=classification_report(pred,Y_test)
cm=confusion_matrix(pred,Y_test)

def check_spam_ham(input_string):
    proceeed_data = preprocess_data(input_string)
    input_tfidf = tfidf_v.transform([proceeed_data])
    return "It's a SPAM message" if nb.predict(input_tfidf) == 1  else "It's a HAM message"

input_string = "HI NLP, LETS MEET ON MONDAY"
print(check_spam_ham(input_string))
