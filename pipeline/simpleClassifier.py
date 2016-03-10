__author__ = 'Haohan Wang'

import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

from sklearn import cross_validation
from sklearn.cross_validation import KFold

from featureExtraction.videoFeature import getFeatureTitle

from utitlity.dataLoader import load_data_label


def crossValidation(videoNum, data, labels):
    ind1 = np.where(labels[:, 0] != videoNum)
    ind2 = np.where(labels[:, 0] == videoNum)
    trD = data[ind1[0], :]
    teD = data[ind2[0], :]
    trL = labels[ind1[0], 1]
    teL = labels[ind2[0], 1]
    return (trD, trL), (teD, teL)


def F1_certain_label(trl, prl, posLabel):
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    for (t, p) in zip(trl, prl):
        if t == posLabel:
            if t == p:
                tp += 1
            else:
                fp += 1
        else:
            if t == p:
                tn += 1
            else:
                fn += 1
    pr = tp/(tp+fp)
    re = tp/(tp+fn)
    if pr == 0  or re == 0:
        return 0, pr, re
    return 2/(1/pr+1/re), pr, re


def run():
    clf = SVC(tol=1e-5, C=5)
    # clf = GaussianNB()
    # clf = RandomForestClassifier()
    data, labels = load_data_label('../data/video_features.csv', '../data/labels.csv')
    labels = np.array(labels)[:,1]
    kf = KFold(data.shape[0], n_folds=5)
    scores = []
    for train_index, test_index in kf:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    print np.mean(scores)

def featureSelection():
    title = getFeatureTitle()
    mapping = np.arange(len(title))
    data, labels = load_data_label('../data/video_features.csv', '../data/labels.csv', mapping)
    selector = SelectKBest(chi2, k=2)
    data = data + 100
    # data = (data - mini)/(maxi-mini)
    selector.fit(data, labels[:, 1])
    scores = selector.scores_
    scores /= (np.mean(data, axis=0))
    print np.argsort(scores)


if __name__ == '__main__':
    run()
    # featureSelection()
