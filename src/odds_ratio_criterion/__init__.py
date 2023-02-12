__version__ = '0.2.0'

# PATCH sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree._classes import CRITERIA_CLF
from sklearn.tree._criterion import Criterion
from sklearn.utils._param_validation import Hidden
from sklearn.utils._param_validation import StrOptions

from .odds_ratio_criterion import OddsRatioCriterion  # noqa
from .xnboost import XNBoostClassifier  # noqa

CRITERIA_CLF['odds_ratio'] = OddsRatioCriterion
DecisionTreeClassifier._parameter_constraints["criterion"] = [StrOptions({"gini", "entropy", "log_loss", "odds_ratio"}), Hidden(Criterion)]
