 #importing reuired libraries

import nltk
#nltk.download()
#nltk.download('punkt') #to download sentence tokenizer
import re
from nltk.stem import SnowballStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

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


#================================TOKENIZATION=========================================
#breaking an paragraph into a list of sentences
sentences = nltk.sent_tokenize(input_paragraph)

#converting a paragraph into a list of words.
words = nltk.word_tokenize(input_paragraph, preserve_line=False)

#converting a paragraph into a list of words, it merges the "." with the last word.
words_preserve_line = nltk.word_tokenize(input_paragraph, preserve_line=True)

#================================STEMMING============================================
#https://towardsdatascience.com/stemming-vs-lemmatization-2daddabcb221
port_stemer = PorterStemmer()
stop_words_list = stopwords.words("english")

porter_stemmed_sentence = ""
for sentence in sentences:
    word_list = nltk.word_tokenize(sentence)
    word_list = [port_stemer.stem(word) for word in word_list if word not in stopwords.words("english")]
    porter_stemmed_sentence += " ".join(word_list)


#snowball is better and faster than a porter stemmer
snow_stemer = SnowballStemmer(language='english')

snow_stemmed_sentence = ""
for sentence in sentences:
    word_list = nltk.word_tokenize(sentence)
    word_list = [snow_stemer.stem(word) for word in word_list if word not in stopwords.words("english")]
    snow_stemmed_sentence += " ".join(word_list)

#=================================LEMMATIZATION===========================================
lemmatizer = WordNetLemmatizer()

lemmatized_sentence = ""
for sentence in sentences:
    word_list = nltk.word_tokenize(sentence)
    word_list = [lemmatizer.lemmatize(word) for word in word_list if word not in stopwords.words("english")]
    lemmatized_sentence += " ".join(word_list)

#===============================BAGOFWORDS=====================================================

lemmatizer = WordNetLemmatizer()
input_sentence_list = nltk.sent_tokenize(input_paragraph)
count_vector = CountVectorizer(max_features = 1500)
corpus_list = []

for sentence in input_sentence_list:
    sentence = sentence.lower()
    sentence = re.sub('[^a-zA-Z]',' ', sentence)
    word_list = nltk.word_tokenize(sentence)
    word_list = [lemmatizer.lemmatize(word) for word in word_list if word not in stopwords.words("english")]
    corpus_list.append(" ".join(word_list))

paragraph_vector_bow = count_vector.fit_transform(corpus_list).toarray()

#==============TFIDF( TERM FREUENCY-INVERSE DOCKUMENT FREQUENCY=======================

lemmatizer = WordNetLemmatizer()
input_sentence_list = nltk.sent_tokenize(input_paragraph)
tfidf_vector = TfidfVectorizer(max_features = 15)
corpus_list = []

for sentence in input_sentence_list:
    sentence = sentence.lower()
    sentence = re.sub('[^a-zA-Z]',' ', sentence)
    word_list = nltk.word_tokenize(sentence)
    word_list = [lemmatizer.lemmatize(word) for word in word_list if word not in stopwords.words("english")]
    corpus_list.append(" ".join(word_list))

paragraph_vector_tfidf = tfidf_vector.fit_transform(corpus_list).toarray()
