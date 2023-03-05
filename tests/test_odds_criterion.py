import unittest

import numpy as np
from sklearn.tree import DecisionTreeClassifier

import odds_ratio_criterion  # noqa


class OddsRatioCriterionTestCase(unittest.TestCase):
    # y(X) = X[0] xor X[1] xor X[2]
    X_xor = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    y_xor = [0, 1, 1, 0, 1, 0, 0, 1]

    def test_odds_ratio_xor_fcn(self):
        X = np.array(self.X_xor)
        y = np.array(self.y_xor).T

        model = DecisionTreeClassifier(max_depth=3, criterion="odds_ratio")
        model.fit(X, y)

        assert np.all(model.predict(X) == y)

        model = DecisionTreeClassifier(max_leaf_nodes=5, criterion="odds_ratio")
        model.fit(X, y)

        assert np.sum(model.predict(X) == y) / y.shape[0] >= 6.0 / 8.0
