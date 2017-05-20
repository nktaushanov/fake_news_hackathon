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

def preprocess_text(df_column, tokenize):
    return [tokenize(value) for value in list(df_column)]


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
    article_fit_vector = text.CountVectorizer(analyzer=lambda x: x)
    domain_fit_fector = text.CountVectorizer(analyzer=lambda x: x)

    article_preprocessed = preprocess_text(df['Content'],
            lambda article: tokenizer.article2words(article, flatten=True))
    domain_preprocessed = preprocess_text(df['Content Url'],
            lambda url: utils.get_domain(url))

    articles_vectors = article_fit_vector.fit_transform(article_preprocessed)
    domain_vectors = domain_fit_fector.fit_transform(domain_preprocessed)

    combined_vectors = sparse.hstack((articles_vectors, domain_vectors))

    scores = list(df['fake_news_score'])

    classifier = tree.DecisionTreeClassifier()
    classifier.fit(combined_vectors, scores)

    print classifier.score(combined_vectors, scores)


    eval_df = corpora.train_set('FN_Evaluation_Set_Small.csv')
    eval_article_preprocessed = preprocess_text(eval_df['Content'],
            lambda article: tokenizer.article2words(article, flatten=True))
    eval_domain_preprocessed = preprocess_text(eval_df['Content Url'],
            lambda url: utils.get_domain(url))

    eval_articles_vectors = article_fit_vector.transform(eval_article_preprocessed)
    eval_domain_vectors = domain_fit_fector.transform(eval_domain_preprocessed)
    eval_combined_vectors = sparse.hstack((eval_articles_vectors, eval_domain_vectors))
    eval_scores = list(eval_df['fake_news_score'])
    
    print classifier.score(eval_combined_vectors, eval_scores)





if __name__ == "__main__":
    df = corpora.train_set('FN_Training_Set.csv')
    combined_classifier(df)
