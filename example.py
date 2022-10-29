from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_validate

from load_datafiles import *

from classification_tree import ClassificationTree

# X, y, test = load_titanic()

X, y = load_mushroom()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = ClassificationTree(min_samples=10, max_depth=20)

print(cross_validate(model, X, y, cv=5, verbose=10, n_jobs=-1))