# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.  
"""
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score


twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
print(twenty_train.target)
print(twenty_train.target.size)
print(twenty_train.target_names)

count_vect=CountVectorizer(stop_words='english',lowercase=True)
print(count_vect)

X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts)
print(X_train_counts.shape) 
 
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)


text_clf_svm = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('svmclf-', SGDClassifier(loss='hinge', penalty='l2',
alpha=1e-3, max_iter=5, random_state=30)),])
text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)

predicted_svm = text_clf_svm.predict(twenty_test.data)

print("Accuracy:",accuracy_score(twenty_test.target,predicted_svm))
print(metrics.classification_report(twenty_test.target,predicted_svm,target_names=twenty_test.target_names))
