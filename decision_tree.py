# -*- coding: utf-8 -*-
"""
Created on Tue May 19 19:14:37 2020

@author: DELL
"""

from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score


newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
X_test = newsgroups_test.data
y_train = newsgroups_train.target
y_test = newsgroups_test.target



text_clf = Pipeline([('vect', CountVectorizer(stop_words='english',lowercase=True)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', tree.DecisionTreeClassifier()),
                     ])

text_clf.fit(X_train, y_train)


predicted = text_clf.predict(X_test)
print("Accuracy:",accuracy_score(newsgroups_test.target,predicted))


print(metrics.classification_report(y_test, predicted))