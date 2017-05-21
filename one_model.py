from scipy import sparse
from sklearn import tree
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction import text
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import preprocessing
import numpy as np
import pandas as pd

import bulstem
from classifier import preprocess_text
import corpora
import tokenizer
import utils
import vectorizer
import word2vec


class Classifier(object):
    def __init__(self, classifier, vectorizer, column_name):
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.column_name = column_name

    def fit(self, data):
        self.vectorizer.fit(data)
        X, y = self.vectorizer.transform(data), data[self.column_name]
        self.classifier.fit(X, y)

    def predict(self, data):
        X = self.vectorizer.transform(data)
        return self.classifier.predict(X)


class DecisionFn(object):
    def __init__(self, score):
        self.score = score
        self.vectorizer = text.CountVectorizer(analyzer=lambda x: x)
        self.classifier = LinearSVC()

    def fit(self, df):
        self.vectorizer.fit(self._get_words(df))
        X = self.vectorizer.transform(self._get_words(df))
        y = self.score(df)
        self.classifier.fit(X, y)

    def match(self, df):
        X = self.vectorizer.transform(self._get_words(df))
        return [abs(x) < 0.5 for x in self.classifier.decision_function(X)]

    def _get_words(self, df):
        return [[x] for x in list(df['Content Url'])]


class Model(object):
    def __init__(self, decision_fn, main_classifier, fallback_classifier):
        self.decision_fn = decision_fn
        self.main_classifier = main_classifier
        self.fallback_classifier = fallback_classifier

    def train(self, df):
        # fit all of the above
        self.decision_fn.fit(df)
        self.main_classifier.fit(df)
        self.fallback_classifier.fit(df)
        return self

    def classify(self, df):
        decision_matches = self.decision_fn.match(df)
        main_predicts = self.main_classifier.predict(df)
        fallback_predicts = self.fallback_classifier.predict(df)
        return [main if decision else fallback
                for (decision, main, fallback) in zip(decision_matches, main_predicts, fallback_predicts)]
        # return a list of 1's and 3's


class DecisionTreeVectorizer(object):
    def __init__(self):
        self.article_fit_vector = text.CountVectorizer(analyzer=lambda x: x)
        self.domain_fit_fector = text.CountVectorizer(analyzer=lambda x: x)

    def fit(self, df):
        article_preprocessed = preprocess_text(df['Content Title'],
                lambda article: tokenizer.article2words(article, flatten=True))
        domain_preprocessed = preprocess_text(df['Content Url'],
                lambda url: utils.get_domain(url))

        self.article_fit_vector.fit(article_preprocessed)
        self.domain_fit_fector.fit(domain_preprocessed)

    def transform(self, df):
        article_preprocessed = preprocess_text(df['Content Title'],
                lambda article: tokenizer.article2words(article, flatten=True))
        domain_preprocessed = preprocess_text(df['Content Url'],
                lambda url: utils.get_domain(url))
        articles_vectors = self.article_fit_vector.transform(article_preprocessed)
        domain_vectors = self.domain_fit_fector.transform(domain_preprocessed)
        return sparse.hstack((articles_vectors, domain_vectors))


class NNVectorizer(object):
    def __init__(self):
        self.content_vectorizer = vectorizer.MeanEmbeddingVectorizer(
                word2vec.get_word_vectors())
        self.domain_vectorizer = text.CountVectorizer(analyzer=lambda x: x)

    def fit(self, df):
        domain_words = [utils.get_domain(value) for value in list(df['Content Url'])]
        self.domain_vectorizer.fit(domain_words)

    def transform(self, df):
        tokenize = lambda article: tokenizer.article2words(article, flatten=True)
        articles_words = preprocess_text(df['Content'], tokenize)
        article_vectors = self.content_vectorizer.transform(articles_words)

        domain_words = [utils.get_domain(value) for value in list(df['Content Url'])]
        domain_vectors = self.domain_vectorizer.transform(domain_words)

        return sparse.hstack((domain_vectors, article_vectors))


class OneModel(object):
    def __init__(self, score_column_name):
        decision_fn = DecisionFn(score)

        decision_tree = Classifier(
                DecisionTreeClassifier(), DecisionTreeVectorizer(), score_column_name)

        nn = Classifier(
                MLPClassifier((50, 5), alpha=.01, learning_rate='adaptive'),
                NNVectorizer(),
                score_column_name)

        self.model = Model(decision_fn, nn, decision_tree)

    def train(self, df):
        self.model.train(df)

    def classify(self, df):
        return self.model.classify(df)

