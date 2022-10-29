import random
import numpy as np
import pandas as pd
import pygraphviz
from math import ceil, sqrt


class Node:
    def __init__(self, split_feature=None, thres=None, left=None, right=None, gain=None, classification=None, samples=None, impurity=None):
        # splitting feature, boundary of split (val <= goes left, higher goes right), and classification if leaf node
        self.split_feature = split_feature
        self.thres = thres
        self.left = left
        self.right = right
        self.gain = gain
        self.classification = classification
        self.impurity = impurity
        self.samples = samples

class ClassificationTree():
    def __init__(self, max_depth=10, min_samples=3):
        self.max_depth = max_depth # max_depth = # of splits, not height
        self.min_samples = min_samples
        self.classes = dict()

    def split_dataset(self, X, y, feature, thres):
        # splits dataset into left_X, left_y, right_X, right_y
        df = pd.concat([y, X], axis=1)
        left = df[df[feature] <= thres]
        right = df[df[feature] > thres]
        return left.iloc[:,1:], left.iloc[:,0], right.iloc[:,1:], right.iloc[:,0]

    def gini(self, y):
        res = 1
        for val in y.value_counts():
            res -= (val/len(y))**2
        return res

    def information_gain(self, y, left_y, right_y):
        # uses gini impurity to calc weighted gain
        return self.gini(y) - (self.gini(left_y)*len(left_y)/len(y) + self.gini(right_y)*len(right_y)/len(y))

    def best_split(self, X, y, n_rows, n_cols):
        # X, y, num of samples, number of features (# of X columns)
        split = {'gain':0, 'split_feature':None, 'thres':None}

        # for each feature in X
        # iterate over all unique vals in that col and split by that val
        for feature in X:
            vals_unique = X[feature].unique()
            vals_unique.sort()

            # if it's random, don't loop thru all, randomly pick a few and choose the best
            if self.split_type != "best":
                vals_unique = np.random.choice(vals_unique, ceil(sqrt(len(vals_unique))), replace=False)

            for thres in vals_unique:
                # split by threshold
                left_X, left_y, right_X, right_y = self.split_dataset(X, y, feature, thres)

                # make sure they have smthg
                if len(left_X) == 0 or len(right_X) == 0: continue

                ig_new = self.information_gain(y, left_y, right_y)
                # calculate info gain using the classes in y
                if ig_new > split['gain']:
                    split['split_feature'] = feature

                    split['thres'] = thres
                    split['left_X'] = left_X
                    split['left_y'] = left_y
                    split['right_X'] = right_X
                    split['right_y'] = right_y

                    split['gain'] = ig_new
        return split


    def build_tree(self, X, y, depth):
        # recursively divide df into subtrees until requirements met

        n_samples, n_features = X.shape # (rows = # of samples, cols = # of features)
        if n_samples >= self.min_samples and depth <= self.max_depth:

            split = self.best_split(X, y, n_samples, n_features)

            if split['gain'] > 0:
                # split into left and right child nodes
                left = self.build_tree(split["left_X"], split['left_y'], depth+1)
                right = self.build_tree(split["right_X"],split['right_y'],depth+1)

                # if not left child and right child both leaf nodes with same classification
                if left.classification==None or left.classification != None and right.classification != left.classification:
                    return Node(split['split_feature'],split['thres'],left,right,split['gain'],samples=y.value_counts().values)
                # otherwise just make this a leaf node with same label as children

        # if doesn't meet the requirements OR doesn't gain from splitting, become the LEAF
        # classification, sample distribution, and gini impurity
        label = y.mode().iloc[0]
        self.classes[label] = 0
        return Node(classification=(label), samples=y.value_counts(), impurity=self.gini(y))

    # all vals must be numbers
    # well actually all panda DataFrames
    def fit(self, X, y, split_type="best"):
        # split_type either loop thru all, or randomly go sqrt(unique) and choose best one
        self.split_type = split_type

        self.y_dtype = y.dtype
        self.root = self.build_tree(X, y, 1)

    def classify(self, row, node):
        # recursively traverse the tree to find class
        if node.classification == None:
            if row[node.split_feature] <= node.thres: return self.classify(row, node.left)
            else: return self.classify(row, node.right)

        else:
            return node.classification

    def predict(self, X):
        preds = pd.Series(index=X.index, dtype=self.y_dtype)
        for index, row in X.iterrows():
            preds[index] = self.classify(row, self.root)
        return preds

    def score(self, X, y, sample_weights=None):
        preds = self.predict(X)
        return (preds==y).sum()/len(y)

    def get_params(self, deep=False):
        return {'min_samples':self.min_samples, 'max_depth':self.max_depth}

    def set_params(self, min_samples=2, max_depth=2):
        self.min_samples = min_samples
        self.max_depth=max_depth

    def _visualize(self, node, i):
        if node.classification != None:
            label = f"{node.classification}\n{round(node.impurity, 3)}\n{node.samples.values}"
            self.g.add_node(i, label=label, style='filled', fillcolor=self.classes[node.classification])
        else:
            label = f"{node.split_feature} <= {node.thres}\n{node.samples}"
            self.g.add_node(i, label=label)

            if node.left:
                self._visualize(node.left, i*2)
                self.g.add_edge(i, i*2)
            if node.right:
                self._visualize(node.right, i*2+1)
                self.g.add_edge(i, i*2+1)


    def visualize(self, filename:str):
        self.g = pygraphviz.AGraph(directed=True)
        self.g.node_attr['shape'] = "box"

        # colour scheme
        self.g.node_attr['colorscheme']='accent8'
        for c in enumerate(self.classes):
            self.classes[c[1]] = c[0]+1

        self._visualize(self.root, 1)

        self.g.layout(prog='dot')
        self.g.draw(filename)