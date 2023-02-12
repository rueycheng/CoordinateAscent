from __future__ import print_function, division

import argparse

from sklearn.datasets import load_svmlight_file

from coordinate_ascent import CoordinateAscent
from metrics import DCGScorer, NDCGScorer
from utils import load_docno, print_trec_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-nv', '--no-validation', action='store_true',
                        help='do not use validation data')
    parser.add_argument('--verbose', action='store_true',
                        help='show verbose output')
    parser.add_argument('-o', '--output-file', metavar='FILE',
                        help='write TREC run output to FILE')
    parser.add_argument('train_file')
    parser.add_argument('valid_file')
    parser.add_argument('test_file')
    args = parser.parse_args()

    X, y, qid = load_svmlight_file(args.train_file, query_id=True)
    X_test, y_test, qid_test = load_svmlight_file(args.test_file, query_id=True)

    model = CoordinateAscent(n_restarts=1,
                             max_iter=25,
                             verbose=args.verbose,
                             scorer=NDCGScorer(k=10, idcg_cache={}))

    if args.no_validation or args.valid_file == '':
        model.fit(X, y, qid)
    else:
        X_valid, y_valid, qid_valid = load_svmlight_file(args.valid_file, query_id=True)
        model.fit(X, y, qid, X_valid, y_valid, qid_valid)
    pred = model.predict(X_test, qid_test)

    for k in (1, 2, 3, 4, 5, 10, 20):
        score = NDCGScorer(k=k)(y_test, pred, qid_test).mean()
        print('nDCG@{}\t{}'.format(k, score))

    if args.output_file:
        docno = load_docno(args.test_file, letor=True)
        print_trec_run(qid_test, docno, pred, output=open(args.output_file, 'wb'))


if __name__ == '__main__':
    main()
