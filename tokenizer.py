# -*- coding: utf-8 -*-

import os
import re

from nltk.tokenize import sent_tokenize, word_tokenize

import corpora
import utils

STEMMING_RULES_FILE = utils.get_project_file_path(
        'resources', 'stemming_rules', 'stem_rules_2.txt')

SENTENCE_SPLIT_REGEX = re.compile(u'^(.*[\.?!])([А-Я][а-я].*)$', re.U)

PUNCTUATION = re.compile(u'^[,\.;:\?!-+"•\'-]+$', re.U)


def split_sentences(sentence):
    matches = SENTENCE_SPLIT_REGEX.match(sentence)
    if matches and matches.groups():
        return [split_sentences(matches.group(1)),
                split_sentences(matches.group(2))]
    else:
        return [sentence]


def extract_sentences(document):
    sentence_tokens = sent_tokenize(document)
    sentences = [split_sentences(token) for token in sentence_tokens]
    return utils.flatten(sentences)


def extract_words(sentence):
    return [word for word in word_tokenize(sentence) if not PUNCTUATION.match(word)]


def article2words(article, flatten=False):
    sentence_words = [extract_words(sentence) for sentence in extract_sentences(article)]
    if flatten:
        return utils.flatten(sentence_words)
    else:
        return sentence_words


if __name__ == "__main__":
    import pandas as pd
    df = corpora.main_data()
    content = list(df['content'])
    for s in extract_sentences(content[0]):
        print 'red: '
        utils.print_unicode(extract_words(s))
    pass


