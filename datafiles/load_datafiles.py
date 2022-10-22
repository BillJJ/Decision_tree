import pandas as pd

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
