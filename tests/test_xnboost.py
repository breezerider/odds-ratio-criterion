import unittest

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from odds_ratio_criterion import XNBoostClassifier


class XNBoostTestCase(unittest.TestCase):
    X_xor = [
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0],
        [1, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
        [1, 1, 1, 1],
    ]

    y_xor = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]

    X_fcn = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]

    y_fcn = [0, 1, 0, 1, 0, 1, 1, 0]

    def test_xnboost_classifier_learning_xor_fcn(self):
        X = np.array(self.X_xor)
        y = np.array(self.y_xor).T

        model = XNBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=4, criterion="odds_ratio"), n_estimators=1, loss_exponent=4, random_state=42
        )
        model.fit(X, y)

        assert np.all(model.predict(X) == y)

    def test_xnboost_classifier_learning_fcn(self):
        X = np.array(self.X_fcn)
        y = np.array(self.y_fcn).T

        model = XNBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=2, criterion="odds_ratio"), n_estimators=3, loss_exponent=4, random_state=42
        )

        model.fit(X, y)

        assert np.all(model.predict(X) == y)

    def test_xnboost_classifier_learning_fail(self):
        X = np.array(self.X_xor)
        y = np.array(self.y_xor).T

        model = XNBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, criterion="odds_ratio"), n_estimators=1, loss_exponent=4, random_state=42
        )
        with self.assertRaises(ValueError) as context:
            model.fit(X, y)

        self.assertTrue(
            'BaseClassifier in XNBoostClassifier ensemble is worse than random, ensemble can not be fit.' in str(context.exception)
        )
