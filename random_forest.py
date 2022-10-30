import numpy
import numpy as np
import math
import pandas as pd

from classification_tree import ClassificationTree

# a bunch of classification trees
# trained on different subsets of training data (sampled with replacement)
# maybe will limit amount of features each tree can use in the future
class ClassificationForest():
    def __init__(self, n_estimators=30, max_depth=10, sample_sizes=0.4):
        self.n_estimators=n_estimators
        self.max_depth=max_depth
        self.sample_sizes=sample_sizes

    def split_dataset(self, X, y, test_size):
        df = pd.concat([X,y],axis=1)
        df = df.sample(frac=test_size)
        return df.iloc[:,:-1], df.iloc[:,-1]

    def fit(self,X,y):
        self.trees = []
        for i in range(self.n_estimators):
            X_sample, y_sample = self.split_dataset(X, y, self.sample_sizes)
            model = ClassificationTree(max_depth=self.max_depth)
            model.fit(X_sample, y_sample,split_type="random")
            self.trees.append(model)

    def predict(self, X):
        preds = {}
        for i in range(self.n_estimators):
            preds[i] = self.trees[i].predict(X)

        preds = pd.DataFrame(preds, index=X.index)
        preds = preds.mode(axis=1).iloc[:,0]

        return preds


    def score(self, X, y):
        preds = self.predict(X)
        return (y==preds).sum()/len(y)