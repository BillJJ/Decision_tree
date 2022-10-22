import pandas as pd
from datafiles.load_datafiles import *
from sklearn.metrics import accuracy_score

from classification_tree import ClassificationTree
from sklearn.model_selection import train_test_split

X, y, test = load_titanic()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = ClassificationTree(min_samples=100, max_depth=4)
model.fit(X, y)
preds = model.predict(test)
preds = list(map(int, preds))

output = pd.DataFrame({'PassengerId': test.index+892, 'Survived':preds})
output.to_csv('datafiles/titanic/submission.csv', index=False)