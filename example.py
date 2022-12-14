import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pygraphviz
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from matplotlib import pyplot as plt

from load_datafiles import *

from classification_tree import ClassificationTree
from random_forest import ClassificationForest

X, y, test, test_index = load_spaceship_titanic()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# model = ClassificationTree(min_samples=12,max_depth=20)
# model.fit(X_train,y_train)
# print(model.score(X_test, y_test))
# model.visualize('datafiles/spaceship-titanic/sample_tree.png')

model = ClassificationForest(n_estimators=30, max_depth=12)
model.fit(X, y)
# print(model.score(X_test, y_test))

preds = model.predict(test)
preds = preds.tolist()
for i in range(len(preds)):
    if preds[i]: preds[i] = 'True'
    else: preds[i] = 'False'

output = pd.DataFrame({'PassengerId':test_index, 'Transported':preds})
output.to_csv('datafiles/spaceship-titanic/submission.csv',index=False)