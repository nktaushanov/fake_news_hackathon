#!/usr/bin/python
# -*- coding: utf-8 -*-

import io
import re

vocals_re = re.compile(u'[^аъоуеияю]*[аъоуеияю]', re.U)
rule_re = re.compile(u'([а-я]+)\s==>\s([а-я]+)\s([0-9]+)', re.U)

class BulStemmer:
    def __init__(self, rules_file, stem_boundary=1):
        self.stem_boundary = stem_boundary
        self.stemming_rules = self._load_rules(rules_file)

    def _load_rules(self, filename):
        stemming_rules = {}
        with io.open(filename, 'r', encoding='utf8') as f:
            for line in f:
                match_result = rule_re.match(line)

                if (match_result and match_result.groups()
                        and len(match_result.groups()) == 3):
                    if int(match_result.group(3)) > self.stem_boundary:
                        stemming_rules[match_result.group(1)] = (
                                match_result.group(2))
        return stemming_rules

    def stem(self, word):
        matched_vocals = vocals_re.match(word)
        if not matched_vocals:
            return word

        for i in xrange(matched_vocals.end() + 1 , len(word)):
            suffix = word[i:]
            if suffix in self.stemming_rules:
                return word[:i] + self.stemming_rules[suffix]

        return word

