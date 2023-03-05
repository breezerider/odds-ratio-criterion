import unittest

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from odds_ratio_criterion import XNBoostClassifier


class XNBoostTestCase(unittest.TestCase):
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

    def test_xnboost_classifier_learning_ensemble(self):
        X = np.array(self.X_xor)
        y = np.array(self.y_xor).T

        model = XNBoostClassifier(
            estimator=DecisionTreeClassifier(max_leaf_nodes=5, criterion="odds_ratio"),
            n_estimators=5,
            loss_exponent=4,
            random_state=42,
        )

        model.fit(X, y)

        assert np.all(model.predict(X) == y)
        assert len(model.estimators_) == 5

    def test_xnboost_classifier_learning_perfect(self):
        X = np.array(self.X_xor)
        y = np.array(self.y_xor).T

        model = XNBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, criterion="odds_ratio"),
            n_estimators=3,
            loss_exponent=4,
            random_state=42,
        )
        model.fit(X, y)

        assert np.all(model.predict(X) == y)
        assert len(model.estimators_) == 1

    def test_xnboost_classifier_learning_fail(self):
        X = np.array(self.X_xor)
        y = np.array(self.y_xor).T

        model = XNBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, criterion="odds_ratio"),
            n_estimators=1,
            loss_exponent=4,
            random_state=42,
        )
        with self.assertRaises(ValueError) as context:
            model.fit(X, y)

        self.assertTrue(
            'BaseClassifier in XNBoostClassifier ensemble is worse than random, ensemble can not be fit.'
            in str(context.exception)
        )
