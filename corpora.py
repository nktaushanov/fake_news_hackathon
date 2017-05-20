import pandas as pd
import utils

def train_set(filename):
    df = pd.read_csv(utils.get_project_file_path('data', filename), encoding='utf8')
    df['is_basestring'] = df['Content'].apply(lambda c: isinstance(c, basestring))
    df = df[df['is_basestring'] == True]
    return df.drop('is_basestring', axis=1)

def main_data():
    df = pd.read_csv(utils.get_project_file_path('resources', 'big', 'main_data_fake_news.csv'), encoding='utf8')
    df['is_basestring'] = df['content'].apply(lambda c: isinstance(c, basestring))
    df = df[df['is_basestring'] == True]
    return df.drop('is_basestring', axis=1)
