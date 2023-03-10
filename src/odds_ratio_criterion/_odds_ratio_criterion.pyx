# cython: linetrace=True, language_level=3str, cdivision=True, boundscheck=False, wraparound=False

import numpy as np

cimport numpy as np

np.import_array()

from numpy.math cimport INFINITY
from sklearn.tree._criterion cimport ClassificationCriterion


cdef class OddsRatioCriterion(ClassificationCriterion):
  r"""Probability ratio based classification tree splitting criterion.
  Designed for binary classification. Assume class count for majority
  and minority class are :math:`N_{maj}` and :math:`N_{min}`, respectively.
  Odds of a node in the split:

      :math:`odds_{c} = \frac{N^{c}_{maj}+N^{o}_{maj}}{N^{c}_{min}+N^{o}_{min}}`

  where :math:`N^{c,o}_{.}` represent counts for current and other leaf node.
  During split evaluation odds for left and right nodes are considered:

      :math:`odds = \min \lbrace odds_{l} \cdot odds_{r} \rbrace`

  The splits are then ranked by :math:`odds` value.
  """

  cdef void odds(self,
                 double *odds_l,
                 double *odds_r) nogil:
    """Compute the odds of a given split in the tree.
    The ratio is relative to the majority class in each of the child nodes.
    """

    cdef double[:, ::1] sum_left = self.sum_left
    cdef double[:, ::1] sum_right = self.sum_right

    if (sum_left[0, 0] + sum_right[0, 0] != 0.0) and (sum_left[0, 1] + sum_right[0, 1] != 0.0):
      # left node
      if(sum_left[0, 0] > sum_left[0, 1]):
        odds_l[0] = (sum_left[0, 0] + sum_right[0, 1]) / (sum_left[0, 1] + sum_right[0, 0])
      else:
        odds_l[0] = (sum_left[0, 1] + sum_right[0, 0]) / (sum_left[0, 0] + sum_right[0, 1])
      # right node
      if(sum_right[0, 0] > sum_right[0, 1]):
        odds_r[0] = (sum_right[0, 0] + sum_left[0, 1]) / (sum_right[0, 1] + sum_left[0, 0])
      else:
        odds_r[0] = (sum_right[0, 1] + sum_left[0, 0]) / (sum_right[0, 0] + sum_left[0, 1])

  cdef double proxy_impurity_improvement(self) nogil:
    cdef double impurity_left
    cdef double impurity_right

    self.children_impurity(&impurity_left, &impurity_right)

    return min(impurity_left, impurity_right)

  cdef double impurity_improvement(self,
                                   double impurity_parent,
                                   double impurity_left,
                                   double impurity_right) nogil:
    """Compute the impurity impurity_improvement.
    Odds in even splits are distributed evenly between left and right nodes.
    To penalize for uneven splitting, choose the lowest of leaf node values.
    """
    return min(impurity_left, impurity_right)

  cdef double node_impurity(self) nogil:
    """Evaluate the impurity of the current node.
    Odds ratio has no practical sense for a node, hence return a pre-set value.
    """
    return 1.0

  cdef void children_impurity(self,
                              double* impurity_left,
                              double* impurity_right) nogil:
    """Evaluate the impurity in children nodes.
    i.e. the impurity of the left child (samples[start:pos]) and the
    impurity the right child (samples[pos:end]).
    Parameters
    ----------
    impurity_left : double pointer
        The memory address to save the impurity of the left node
    impurity_right : double pointer
        The memory address to save the impurity of the right node
    """
    cdef double odds_l = -INFINITY
    cdef double odds_r = -INFINITY

    self.odds(&odds_l, &odds_r)

    impurity_left[0]  = odds_l / self.n_outputs
    impurity_right[0] = odds_r / self.n_outputs
