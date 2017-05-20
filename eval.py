# -*- coding: utf-8 -*-

import os

from sklearn import tree
from sklearn.feature_extraction import text
from scipy import sparse
import numpy as np
import pandas as pd

import corpora
import tokenizer
import utils
import vectorizer
import classifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


CLASSIFIERS = {
    # "Nearest Neighbors": KNeighborsClassifier(3),
    "Linear SVM": SVC(kernel="linear", C=0.025),
    # "RBF SVM": SVC(gamma=2, C=1),
    # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    "Decision Tree": DecisionTreeClassifier(max_depth=10),
    "Random Forest": RandomForestClassifier(max_depth=10, n_estimators=10),
    "Neural Net": MLPClassifier((100, 50, 5), alpha=1, learning_rate='adaptive'),
    # "AdaBoost": AdaBoostClassifier(),
    # "Naive Bayes": GaussianNB(),
    # "QDA": QuadraticDiscriminantAnalysis()
    }


def evaluate(data, labels, classifiers):
    # X_train, X_test, y_train, y_test = train_test_split(
    #         data, labels, test_size=.2, random_state=0)
    for name, classifier in classifiers.iteritems():
        # classifier.fit(X_train, y_train)
        print 'Score for ', name, ': ', np.mean(cross_val_score(classifier, data, labels, cv=10))


def main():
    df = corpora.train_set()
    articles_vectors = classifier.vectorize_column(df['Content Title'],
            text.TfidfVectorizer(analyzer=lambda x: x, max_features=200),
            lambda article: tokenizer.article2words(article, flatten=True))

    domain_vectors = classifier.vectorize_column(df['Content Url'],
            text.CountVectorizer(analyzer=lambda x: x),
            lambda url: utils.get_domain(url))

    combined_vectors = sparse.hstack((articles_vectors, domain_vectors))

    scores = list(df['fake_news_score'])
    evaluate(combined_vectors, scores, CLASSIFIERS)


if __name__ == "__main__":
    main()
