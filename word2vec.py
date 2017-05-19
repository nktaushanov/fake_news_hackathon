# -*- coding: utf-8 -*-

from gensim.models import keyedvectors

import utils

def get_word_vectors():
    return keyedvectors.KeyedVectors.load_word2vec_format(
        utils.get_project_file_path('resources', 'bg.bin'),
        binary=True,
        encoding='utf8',
        unicode_errors='ignore')
