# -*- coding: utf-8 -*-
"""
Created on Sun May 30 22:28:08 2021

@author: akash
"""

import nltk
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
df = pd.read_csv('smsspam.csv',names = ['label','messages'])

corpus = []
wn = WordNetLemmatizer()
x = df['messages']
for i in range(len(x)):
    spam = re.sub('[^a-zA-Z]',' ',x[i])
    spam = spam.lower()
    spam = spam.split()
    spam = [wn.lemmatize(y) for y in spam if y not in set(stopwords.words('english'))]
    spam = ' '.join(spam)
    corpus.append(spam)
from sklearn.feature_extraction.text import TfidfVectorizer
tfid = TfidfVectorizer(max_features=(5000))
z = tfid.fit_transform(corpus).toarray()

y = pd.get_dummies(df['label'])
y = y.iloc[:,1].values

#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(z,y,test_size = 0.2,random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train,y_train)
y_pred = spam_detect_model.predict((x_test))
c = spam_detect_model.score(x_test,y_test) 

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)


