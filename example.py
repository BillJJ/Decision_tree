from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, cross_validate
import pygraphviz

from load_datafiles import *

from classification_tree import ClassificationTree

X, y = load_breast()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = ClassificationTree(min_samples=5, max_depth=20)

model.fit(X_train, y_train, split_type="random")

print(model.score(X_test, y_test))

model.visualize('tree_visual.png')