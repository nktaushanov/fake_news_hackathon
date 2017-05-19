# -*- coding: utf-8 -*-

import os
import re

from nltk.tokenize import sent_tokenize, word_tokenize

import utils

STEMMING_RULES_FILE = utils.get_project_file_path(
        'resources', 'stemming_rules', 'stem_rules_2.txt')

SENTENCE_SPLIT_REGEX = re.compile(u'^(.*[\.?!])([А-Я][а-я].*)$', re.U)

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
    return word_tokenize(sentence)


def article2words(artice):
    return [extract_words(sentence) for sentence in extract_sentences(article)]

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('data/main_data_fake_news.csv', encoding='utf8')
    content = list(df['content'])
    for s in extract_sentences(content[0]):
        print 'red: '
        utils.print_unicode(extract_words(s))
    pass


