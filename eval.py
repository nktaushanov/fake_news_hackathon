# -*- coding: utf-8 -*-

import os

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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd

import bulstem
import classifier
import corpora
import tokenizer
import utils
import vectorizer
import word2vec


STEMMING_RULES_FILE = utils.get_project_file_path(
        'resources', 'stemming_rules', 'stem_rules_2.txt')


CLASSIFIERS = {
    # "Nearest Neighbors": KNeighborsClassifier(3),
    # "Linear SVM": SVC(kernel="linear", C=0.025),
    # "RBF SVM": SVC(gamma=2, C=1),
    # "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    # "Decision Tree": DecisionTreeClassifier(max_depth=10),
    # "Random Forest": RandomForestClassifier(max_depth=10, n_estimators=10),
    "Neural Net": MLPClassifier((50, 5), alpha=.01, learning_rate='adaptive'),
    # "AdaBoost": AdaBoostClassifier(),
    # "Naive Bayes": GaussianNB(),
    # "QDA": QuadraticDiscriminantAnalysis()
    }


def cross_evaluate(data, labels, classifiers):
    # X_train, X_test, y_train, y_test = train_test_split(
    #         data, labels, test_size=.2, random_state=0)
    for name, classifier in classifiers.iteritems():
        # classifier.fit(X_train, y_train)
        print 'Score for ', name, ': ', np.mean(cross_val_score(classifier, data, labels, cv=10))


def evaluate(X_train, y_train, X_test, y_test, classifiers):
    for name, classifier in classifiers.iteritems():
        classifier.fit(X_train, y_train)
        print 'Score for ', name, ': ', classifier.score(X_test, y_test)


def run_cross_evaluation():
    df = corpora.train_set('FN_Training_Set.csv')
    main_set = corpora.main_data()

    tokenize = lambda article: tokenizer.article2words(article, flatten=True)
    main_set_words = [tokenize(article) for article in list(main_set['content'])]
    articles_words = [tokenize(article) for article in list(df['Content Title'])]
    # content_vectorizer = text.TfidfVectorizer(analyzer=lambda x: x, max_features=200)

    # content_vectorizer = vectorizer.MeanEmbeddingVectorizer(word2vec.get_word_vectors())
    stemmer = bulstem.BulStemmer(STEMMING_RULES_FILE)
    word_transform = lambda word: word #stemmer.stem(word.lower())
    content_vectorizer = vectorizer.TfidfEmbeddingVectorizer(
            word2vec.get_word_vectors(), tfidf_word_transform=word_transform)
    content_vectorizer.fit(main_set_words)

    titles = [tokenize(title) for title in list(df['Content Title'])]
    articles_vectors = content_vectorizer.transform(titles)

    domain_vectors = classifier.vectorize_column(df['Content Url'],
            text.CountVectorizer(analyzer=lambda x: x, binary=True),
            lambda url: utils.get_domain(url))

    combined_vectors = sparse.hstack((articles_vectors, domain_vectors))

    scores = list(df['fake_news_score'])
    cross_evaluate(combined_vectors, scores, CLASSIFIERS)


def run_test():
    train = corpora.train_set('FN_Training_Set.csv')
    test = corpora.train_set('FN_Evaluation_Set_Only.csv')
    main_set = corpora.main_data()

    # train = train.append(train[train.fake_news_score == 1])
    train['fake_news_score'] = (train.fake_news_score == 1).apply(lambda c: 1 if c else -1)
    test['fake_news_score'] = (test.fake_news_score == 1).apply(lambda c: 1 if c else -1)

    tokenize = lambda article: tokenizer.article2words(article, flatten=True)
    # stemmer = bulstem.BulStemmer(STEMMING_RULES_FILE)
    word_transform = lambda word: word #stemmer.stem(word.lower())
    content_vectorizer = vectorizer.MeanEmbeddingVectorizer(
    # content_vectorizer = vectorizer.TfidfEmbeddingVectorizer(
            word2vec.get_word_vectors())
            # word2vec.get_word_vectors(), tfidf_word_transform=word_transform)

    # main_set_words = [tokenize(article) for article in list(main_set['content'])]
    # content_vectorizer.fit(main_set_words)

    train_articles_words = classifier.preprocess_text(train['Content'], tokenize)
    train_article_vectors = content_vectorizer.transform(train_articles_words)
    test_article_vectors = content_vectorizer.transform(
            classifier.preprocess_text(test['Content'], tokenize))


    domain_vectorizer = text.CountVectorizer(analyzer=lambda x: x)
    train_domain_vectors = classifier.vectorize_column(
            train['Content Url'], domain_vectorizer, utils.get_domain)
    test_domain_vectors = domain_vectorizer.transform(
            classifier.preprocess_text(test['Content Url'], utils.get_domain))


    train_combined_vectors = sparse.hstack((train_article_vectors, train_domain_vectors))
    test_combined_vectors = sparse.hstack((test_article_vectors, test_domain_vectors))

    train_scores = list(train['fake_news_score'])
    test_scores = list(test['fake_news_score'])
    print 'Running against training sets (facepalm)'
    evaluate(train_combined_vectors, train_scores, train_combined_vectors, train_scores, CLASSIFIERS)
    print 'Running against evaluation set'
    evaluate(train_combined_vectors, train_scores, test_combined_vectors, test_scores, CLASSIFIERS)


def main():
    run_test()


if __name__ == "__main__":
    main()
