
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from odds_ratio_criterion import XNBoostClassifier


def test_xnboost_classifier():
    X_train = np.array([0, 1]).reshape(-1, 1)
    y_train = np.array([1, 0]).T

    X_test = np.array([1, 0]).reshape(-1, 1)
    y_test = np.array([0, 1]).T

    for criterion in ["gini", "entropy", "log_loss", "odds_ratio"]:
        model = XNBoostClassifier(DecisionTreeClassifier(max_depth=1, criterion=criterion), n_estimators=1, loss_exponent=4, random_state=42)
        model.fit(X_train, y_train)

        assert np.all(model.predict(X_test) == y_test)
