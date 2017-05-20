import pandas as pd
import utils

def train_set():
    df = pd.read_csv(utils.get_project_file_path('data', 'FN_Training_Set.csv'), encoding='utf8')
    df['is_basestring'] = df['Content'].apply(lambda c: isinstance(c, basestring))
    df = df[df['is_basestring'] == True]
    return df.drop('is_basestring', axis=1)
