#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 20:43:25 2021

@author: subham
"""

#importing reuired libraries
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec

input_paragraph = '''Machine learning (ML) is the study of computer algorithms that
                    improve automatically through experience and by the use of data.
                    It is seen as a part of artificial intelligence. Machine learning
                    algorithms build a model based on sample data, known as "training
                    data", in order to make predictions or decisions without being
                    explicitly programmed to do so. Machine learning algorithms
                    are used in a wide variety of applications, such as in medicine,
                    email filtering, speech recognition, and computer vision, where
                    it is difficult or unfeasible to develop conventional algorithms
                    to perform the needed tasks.'''

#preprocess the data from creating the model
lemmatizer =WordNetLemmatizer()
def preprocess_data(input_para):
    output_string_list = []
    input_string = nltk.sent_tokenize(input_para)
    for items in input_string:
        review = re.sub('[^a-zA-Z]', ' ',items)
        review = review.lower()
        review = review.split()
        review = [lemmatizer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        output_string_list.append(review)
    return output_string_list


processed_data = preprocess_data(input_paragraph)

#create the mddel
model = Word2Vec(processed_data, min_count =1)

# get all the vocabulary in the model
words = model.wv.vocab

#get the vector for the word make
vector = model.wv["make"]

# find all the words that are releveant to the work make.
similar = model.wv.most_similar("make")
