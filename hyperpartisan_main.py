##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

import argparse
import sys
import HyperpartisanNewsReader as hpnr
from sklearn.naive_bayes import MultinomialNB as NB
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC
from xgboost import XGBClassifier as XGB
from autosklearn.classification import AutoSklearnClassifier as ASC
from bayes_opt import BayesianOptimization as OPT

def do_experiment(args):
    labeler = hpnr.BinaryLabel()
    vocab = hpnr.HNVocab(args.vocabulary, args.vocab_size, args.stop_words)
    bow = hpnr.BagOfWordsFeatures(vocab)
    y = labeler.process(args.labels, args.train_size)
    X, ids = bow.process(args.training, args.train_size)

    xgb = XGB()
    clf = GridSearchCV(xgb,{'max_depth':[3,6,12],
                            'eta':[0.01, 0.1],
                            'gamma':[0.1,1],
                            'colsample_bytree':[0.5,1]}, verbose=1, n_jobs=32)
    clf.fit(X, y)
    with open('opt400knotes.txt','w') as f:
      f.write(clf.cv_results_)
      f.write(clf.best_score_)
      f.write(clf.best_params_)


    if args.test_data != None:
        X_test, ids = bow.process(args.test_data, args.test_size)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)
    else:
        y_pred = cross_val_predict(clf, X, y, cv=args.xvalidate)
        y_prob = cross_val_predict(clf, X, y, cv=args.xvalidate, method='predict_proba')
        print(y_pred)
        print(y_prob)
    for i in range(len(y_pred)):
        clas = "true" if y_pred[i] == 1 else "false"
        pred = max(y_prob[i][0], y_prob[i][1])
        args.output_file.write(str(ids[i]) + " " + clas + " " + str(pred)+"\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("training", type=argparse.FileType('rb'), help="Training articles")
    parser.add_argument("labels", type=argparse.FileType('rb'), help="Training article labels")
    parser.add_argument("vocabulary", type=argparse.FileType('r'), help="Vocabulary")
    parser.add_argument("-o", "--output_file", type=argparse.FileType('w'), default=sys.stdout, help="Write predictions to FILE", metavar="FILE")
    parser.add_argument("-s", "--stop_words", type=int, metavar="N", help="Exclude the top N words as stop words", default=None)
    parser.add_argument("-v", "--vocab_size", type=int, metavar="N", help="Only count the top N words from the vocab file (after stop words)", default=None)
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances.", default=None)
    parser.add_argument("--test_size", type=int, metavar="N", help="Only test on the first N instances.", default=None)

    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("-t", "--test_data", type=argparse.FileType('rb'), metavar="FILE")
    eval_group.add_argument("-x", "--xvalidate", type=int)

    args = parser.parse_args()
    do_experiment(args)

    # for fp in (args.output_file, args.training, args.labels, args.vocabulary): fp.close()
