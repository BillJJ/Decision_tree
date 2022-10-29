import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

pd.set_option('display.max_columns', None)

"""
loading data into X, y, and x_test (if applicable);
"""

def load_titanic():
    train = pd.read_csv('datafiles/titanic/train.csv')
    test = pd.read_csv('datafiles/titanic/test.csv')

    y = train['Survived']
    X = train.drop('Survived', axis=1)

    X['Sex'] = (X['Sex']=='male').astype(int)
    X = X[['Sex', 'SibSp', 'Parch']]

    test['Sex'] = (test['Sex']=='male').astype(int)
    test = test[['Sex', 'SibSp', 'Parch']]

    return X, y, test

def load_mushroom():
    df = pd.read_csv('datafiles/mushroom/mushrooms.csv')
    enc = OrdinalEncoder()
    columns = df.columns
    df = pd.DataFrame(enc.fit_transform(df))
    df.columns = columns

    return df.drop('class', axis=1), df['class']

load_mushroom()