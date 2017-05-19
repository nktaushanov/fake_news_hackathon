# -*- coding: utf-8 -*-

import bulstem
import utils


STEMMING_RULES_FILE = utils.get_project_file_path(
        'resources', 'stemming_rules', 'stem_rules_2.txt'
)


utils.print_unicode(bulstem.BulStemmer(STEMMING_RULES_FILE).stem(u'хората'))
