# CoordinateAscent

This is a python implementation of the Coordinate Ascent algorithm (Metzler and Croft, 2007).

The structure of the code follows closely to the scikit-learn style, but still there are some
differences in the estimator/metrics API (e.g. `fit()` method takes three arguments `X`, `y`,
and `qid` rather than just two).

Four ranking metrics are implemented: P@k, AP, DCG@k, and nDCG@k
(in both `trec_eval` and Burges et al. versions). 

## Dependencies

* `numpy`
* `scikit-learn`

## Usage

The following code will run Coordinate Ascent for 2 random restarts and 25
iterations for line search in each dimension, optimizing for NDCG@10.

```
from coordinate_ascent import CoordinateAscent
from metrics import NDCGScorer

scorer = NDCGScorer(k=10, idcg_cache={})
model = CoordinateAscent(n_restarts=2, max_iter=25, verbose=True, scorer=scorer).fit(X, y, qid)
pred = model.predict(X_test, qid_test)
print scorer(y_test, pred, qid_test).mean()
```

Note that, when setting `idcg_cache` to some dict object, NDCGScorer will cache
the ideal DCG score (wrt the qid) to speed up training.  Be sure to use
separate scorers if there is an overlap between training/test qids.

See [test.py](test.py) for more advanced examples.

## References

Metzler and Croft. Linear feature-based models for information retrieval.
_Information Retrieval_, 10(3): 257&ndash;274, 2007.
