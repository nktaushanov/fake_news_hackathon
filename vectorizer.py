# -*- coding: utf-8 -*-

import os
import re
from collections import defaultdict

import numpy as np
from gensim.models import keyedvectors
from sklearn.feature_extraction import text

import utils
import tokenizer

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = word2vec.vector_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = word2vec.vector_size

    def fit(self, X, y=None):
        tfidf = text.TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/main_data_fake_news.csv', encoding='utf8')
    content = list(df['content'])
    word_vectors = keyedvectors.KeyedVectors.load_word2vec_format(
            utils.get_project_file_path('data/bg.bin'),
            binary=True,
            encoding='utf8',
            unicode_errors='ignore')

    article_words = utils.flatten(tokenizer.article2words(content[0]))
    print MeanEmbeddingVectorizer(word_vectors).transform([article_words])
    print TfidfEmbeddingVectorizer(word_vectors).fit([article_words]).transform([article_words])


