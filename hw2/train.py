#!/usr/bin/env python
# -*- coding: utf-8 -*-

import optparser
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def run(X):
    # Train 90%, Dev 10%
    clf1 = svm.LinearSVC()
    pred1 = clf1.fit(X[:23588, :], y_train).predict(X[23588:, :])
    print "LinearSVC: %f" % accuracy_score(pred1, y_dev)
    clf1 = svm.SVC()
    pred1 = clf1.fit(X[:23588, :], y_train).predict(X[23588:, :])
    print "SVC(RBF): %f" % accuracy_score(pred1, y_dev)
    clf1 = RandomForestClassifier(n_estimators=1000)
    pred1 = clf1.fit(X[:23588, :], y_train).predict(X[23588:, :])
    print "RandomForest(1000): %f" % accuracy_score(pred1, y_dev)
