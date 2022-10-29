import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

pd.set_option('display.max_columns', None)

"""
loading data into X_train, X_test, y_train, y_test;
unless there is no y_test (titanic)
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

    X, test_X, y, test_y = train_test_split(df.drop('class', axis=1), df['class'], test_size=0.2)
    return X, test_X, y, test_y

load_mushroom()