# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
import numpy
import sklearn.feature_extraction.text
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score


def classifier(clf, X, y):
    global X_train
    global y_train
    global X_test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    precision = numpy.mean(y_predicted == y_test)
    print(precision)       
    
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
count_vector = sklearn.feature_extraction.text.CountVectorizer()
x = count_vector.fit_transform(twenty_train.data)


tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer(use_idf=True,sublinear_tf=False).fit(x)
TFIDF = tfidf_transformer.transform(x)
#predicted_knn = text_clf_svm.predict(twenty_test.data)
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
Y_predict=knn.predict(X_test)

n_neighbors = 5
weights = 'uniform'

for k in range(1,25):
    print(k)
    clf = KNeighborsClassifier(n_neighbors, weights=weights)
    classifier(clf, TFIDF, twenty_train.target)
print("Accuracy:",accuracy_score(twenty_test.target,Y_predict))
print(metrics.classification_report(twenty_test.target,predicted,target_names=twenty_test.target_names))

