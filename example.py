from sklearn.metrics import accuracy_score

from load_datafiles import *

from classification_tree import ClassificationTree

# X, y, test = load_titanic()

X_train, X_test, y_train, y_test = load_mushroom()

model = ClassificationTree(min_samples=100, max_depth=4)
model.fit(X_train, y_train)
preds = model.predict(X_test)

print(accuracy_score(preds, y_test))