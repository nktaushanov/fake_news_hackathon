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


def vectorize_column(df_column, vectorizer, tokenize):
    all_articles = [tokenize(value) for value in list(df_column)]
    return vectorizer.fit_transform(all_articles)


def content_dicision_tree(df):
    articles_vectors = vectorize_column(df['Content'],
            text.CountVectorizer(analyzer=lambda x: x),
            lambda article: tokenizer.article2words(article, flatten=True))

    # titles = [tokenizer.article2words(title, flatten=True)
    #         for title in list(df['Content Title'])]
    scores = list(df['fake_news_score'])

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(articles_vectors, scores)

    print classifier.score(articles_vectors, scores)


def domain_dicision_tree(df):
    domain_vectors = vectorize_column(df['Content Url'],
            text.CountVectorizer(analyzer=lambda x: x),
            lambda url: utils.get_domain(url))

    # titles = [tokenizer.article2words(title, flatten=True)
    #         for title in list(df['Content Title'])]
    scores = list(df['fake_news_score'])

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(domain_vectors, scores)

    print classifier.score(domain_vectors, scores)


def combined_classifier(df):
    articles_vectors = vectorize_column(df['Content'],
            text.CountVectorizer(analyzer=lambda x: x),
            lambda article: tokenizer.article2words(article, flatten=True))

    domain_vectors = vectorize_column(df['Content Url'],
            text.CountVectorizer(analyzer=lambda x: x),
            lambda url: utils.get_domain(url))

    combined_vectors = sparse.hstack((articles_vectors, domain_vectors))

    scores = list(df['fake_news_score'])

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(combined_vectors, scores)

    print classifier.score(combined_vectors, scores)


if __name__ == "__main__":
    df = corpora.train_set()
    combined_classifier(df)
