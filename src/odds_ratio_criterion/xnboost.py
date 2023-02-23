"""X^N Boost module.

Module contains the :class:`~odds_ratio_criterion.XNBoostClassifier` that
implements adaptive resampling for boosting unweighted ensembles of weak
learners applied to classification problems.
"""

from numbers import Real

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.ensemble._weight_boosting import BaseWeightBoosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._param_validation import Interval
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_is_fitted


class XNBoostClassifier(ClassifierMixin, BaseWeightBoosting):
    """An Adaptive Resampling and Combining (ARC) X^N classifier based on ARC-X4.

    An Adaptive Resampling and Combining X^N classifier is based on the
    ARC-X4 algorithm [1]_ and follows the idea of adaptive resampling
    for boosting unweighted ensembles of weak learners. The sample weights
    are adjusted according to total misclassification over all the boosting
    steps.

    Parameters
    ----------
    estimator : object, default=None
        The base estimator from which the boosted ensemble is built.
        Support for sample weighting is required, as well as proper
        ``classes_`` and ``n_classes_`` attributes. If ``None``, then
        the base estimator is :class:`~sklearn.tree.DecisionTreeClassifier`
        initialized with `max_depth=1`.

    n_estimators : int, default=50
        The maximum number of estimators at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`.

    learning_rate : float, default=1.0
        Weight applied to each classifier at each boosting iteration. A higher
        learning rate increases the contribution of each classifier. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`.

    loss_exponent : float, default=4.0
        The value of power exponent used for computing sample weights based on
        classification results from each of the weak learners in the ensemble.
        Values must be in range `(0.0, inf)`.

    random_state : int, RandomState instance or None, default=None
        Controls the random seed given at each `estimator` at each
        boosting iteration.
        Thus, it is only used when `estimator` exposes a `random_state`.
        Pass an int for reproducible output across multiple function calls.
        See :class:`numpy:numpy.random.RandomState`.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    estimator_errors_ : ndarray of floats
        Classification error for each estimator in the boosted
        ensemble.

    References
    ----------
    .. [1] Leo Breiman; Prediction Games and Arcing Algorithms. Neural Comput 1999; 11 (7): 1493–1517.
    """

    _parameter_constraints: dict = {
        **BaseWeightBoosting._parameter_constraints,
        "loss_exponent": [Interval(Real, 0, None, closed="neither")],
    }

    @_deprecate_positional_args
    def __init__(self, estimator=None, *, n_estimators=50, learning_rate=1.0, loss_exponent=4.0, random_state=None):
        super().__init__(
            estimator=estimator, n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state
        )

        self.loss_exponent = loss_exponent

    def _validate_estimator(self):
        """Check the estimator and set the base_estimator_ attribute."""
        super(BaseWeightBoosting, self)._validate_estimator(default=DecisionTreeClassifier(max_depth=1))

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

        self.misclassifications_ = None

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

        # Stop if the error is at least as bad as random guessing
        if estimator_error >= 1.0 - (1.0 / self.n_classes_):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    'BaseClassifier in XNBoostClassifier ensemble is worse than random, ensemble can not be fit.'
                )
            return None, None, None

        estimator_weight = None

        # sample weights
        misclassifications = np.count_nonzero(self.misclassifications_, axis=0)

        sample_weight = 1.0 + np.power(misclassifications, self.loss_exponent) * self.learning_rate

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
        predictions = self._predictions(X)

        if self.n_classes_ > 1:
            return self.classes_.take(np.median(predictions, axis=1).astype(dtype=int), axis=0)
        else:
            return np.repeat(self.classes_[0], (_num_samples(X)))

    def _predictions(self, X, limit=None):
        """Provide predictions using at most limit members of the ensemble"""
        check_is_fitted(self)
        X = self._check_X(X)

        limit = np.clip(limit, 0, len(self.estimators_)) if limit else len(self.estimators_)

        # Evaluate predictions of all estimators
        return np.array([est.predict(X) for est in self.estimators_[:limit]]).T
