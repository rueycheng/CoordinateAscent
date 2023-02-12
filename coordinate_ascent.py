"""
Coordinate Ascent algorithm

This code is derived from the RankLib implementation https://www.lemurproject.org/ranklib.php

Original paper:
- Metzler and Croft (2007). Linear feature-based models for information retrieval. Information Retrieval, 10(3): 257-274.
  http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5156&rep=rep1&type=pdf
"""
from __future__ import print_function, division

import numpy as np
import sklearn

from sklearn.utils import check_X_y

from metrics import NDCGScorer


class CoordinateAscent(sklearn.base.BaseEstimator):
    """Coordinate Ascent"""

    def __init__(self, n_restarts=5, max_iter=25, tol=0.0001, verbose=False, scorer=None):
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.scorer = scorer

    def fit(self, X, y, qid, X_valid=None, y_valid=None, qid_valid=None):
        """Fit a model to the data"""
        X, y = check_X_y(X, y, 'csr')
        X = X.toarray()

        if X_valid is None:
            X_valid, y_valid, qid_valid = X, y, qid  # noqa
        else:
            X_valid, y_valid = check_X_y(X_valid, y_valid, 'csr')
            X_valid = X_valid.toarray()

        # use nDCG@10 as the default scorer
        if self.scorer is None:
            self.scorer = NDCGScorer(k=10)

        best_score, best_coef = float('-inf'), None

        for restart_no in range(1, self.n_restarts + 1):
            coef = np.ones(X.shape[1], dtype=np.float64) / X.shape[1]
            score = self.scorer(y, np.dot(X, coef), qid).mean()

            n_fails = 0  # count the number of *consecutive* failures
            while n_fails < X.shape[1] - 1:
                for iter_no, fid in enumerate(np.random.permutation(X.shape[1]), 1):
                    best_local_score, best_change = score, None

                    pred = np.dot(X, coef)
                    pred_delta = X[:, fid]
                    stepsize = 0.05 * np.abs(coef[fid]) if coef[fid] != 0 else 0.001

                    change = stepsize
                    for j in range(self.max_iter):
                        new_score = self.scorer(y, pred + change * pred_delta, qid).mean()
                        if new_score > best_local_score:
                            best_local_score, best_change = new_score, change
                        change *= 2

                    if best_change is None:
                        change = stepsize
                        for j in range(self.max_iter):
                            new_score = self.scorer(y, pred - change * pred_delta, qid).mean()
                            if new_score > best_local_score:
                                best_local_score, best_change = new_score, -change
                            change *= 2

                    if best_change is not None:
                        score = best_local_score
                        coef[fid] += best_change
                        coef /= np.abs(coef).sum()  # renormalize the coefficients
                        if self.verbose:
                            print('{}\t{}\t{}\t{}'.format(restart_no, iter_no, fid, score))
                        n_fails = 0
                    else:
                        n_fails += 1

            if score > best_score + self.tol:
                best_score, best_coef = score, coef.copy()

        self.coef_ = best_coef
        return self

    def predict(self, X, qid):
        """Make predictions"""
        return np.dot(X.toarray(), self.coef_)
