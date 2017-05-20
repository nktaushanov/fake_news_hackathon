# -*- coding: utf-8 -*-

import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn import tree

import tokenizer
import utils
import vectorizer



if __name__ == "__main__":
    df = pd.read_csv('data/FN_Training_Set.csv', encoding='utf8')
    df['is_basestring'] = df['Content'].apply(lambda c: isinstance(c, basestring))
    df = df[df['is_basestring'] == True]
    count_vectorizer = text.CountVectorizer(analyzer=lambda x: x)

    tokenize = lambda article: tokenizer.article2words(article, flatten=True)
    all_articles = [tokenize(article) for article in list(df['Content'])]
    articles_vectors = count_vectorizer.fit_transform(all_articles)

    # titles = [tokenizer.article2words(title, flatten=True)
    #         for title in list(df['Content Title'])]
    scores = list(df['fake_news_score'])

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(articles_vectors, scores)

    print classifier.score(articles_vectors, scores)
