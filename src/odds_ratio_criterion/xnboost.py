#!/usr/bin/env python3
# Classifier based on the arc-x4 algorithm from (Breiman, 1998)

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble._weight_boosting import BaseWeightBoosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_is_fitted


class XNBoostClassifier(ClassifierMixin, BaseWeightBoosting):
    """An ARC-X4 classifier.

    An Adaptive Resampling and Combining X^N classifier is based on the algorithm
    proposed in [1] and follows the idea of adaptive resampling for boosting unweighted
    ensembles of weak learners. The sample weights are adjusted according to total
    misclassification over all the boosting steps.


    References
    ----------
    .. [1] Leo Breiman; Prediction Games and Arcing Algorithms. Neural Comput 1999; 11 (7): 1493–1517.
    """

    @_deprecate_positional_args
    def __init__(self, base_estimator=None, *, n_estimators=50, learning_rate=1.0, loss_exponent=4, random_state=None):

        super().__init__(estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)

        self.loss_exponent = 4 if loss_exponent is None else loss_exponent

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(BaseWeightBoosting, self)._validate_estimator(default=DecisionTreeClassifier(max_depth=3))

    def fit(self, X, y, sample_weight=None):
        """Build a boosted regressor from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        y : array-like of shape (n_samples,)
            The target values (real numbers).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
        """
        # Initialize misclassifications
        self.misclassifications_ = np.zeros((self.n_estimators, _num_samples(X)), dtype=bool)

        # Fit
        c = super().fit(X, y, sample_weight)

        self.indexes_ = None
        self.missclassifications_ = None

        # quit()

        return c

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState instance
            The RandomState instance used if the base estimator accepts a
            `random_state` attribute.

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Instances incorrectly classified
        incorrect = y_predict != y

        self.misclassifications_[iboost, :] = incorrect

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in XNBoostClassifier ' 'ensemble is worse than random, ensemble ' 'can not be fit.')
            return None, None, None

        estimator_weight = None

        # sample weights
        misclassifications = np.count_nonzero(self.misclassifications_, axis=0)

        sample_weight = 1.0 + np.power(misclassifications, self.loss_exponent * self.learning_rate)

        return sample_weight, estimator_weight, estimator_error

    def predict(self, X):
        """Predict regression value for X.

        The predicted regression value of an input sample is computed
        as the weighted median prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. COO, DOK, and LIL are converted to CSR.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted regression values.
        """
        # Evaluate predictions of all estimators
        predictions = np.asarray(self.predictions(X))

        if len(self.classes_) > 1:
            return self.classes_.take(np.median(predictions, axis=1).astype(dtype=int), axis=0)
        else:
            return np.repeat(self.classes_[0], (_num_samples(X)))

    def predictions(self, X, limit=None):
        check_is_fitted(self)
        X = self._check_X(X)

        limit = len(self.estimators_) if limit is None else limit

        # Evaluate predictions of all estimators
        return np.array([est.predict(X) for est in self.estimators_[:limit]]).T
