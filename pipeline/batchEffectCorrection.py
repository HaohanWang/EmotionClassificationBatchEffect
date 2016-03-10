__author__ = 'Haohan Wang'

import numpy as np
from utitlity.dataLoader import load_data_label
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score

from LinearMixedModel.LMM import LMM


def run():
    np.random.seed(0)
    clf = LMM()
    data, label = load_data_label('../data/video_features.csv', '../data/labels.csv')
    batch = np.array(label)[:,0].reshape((len(label), 1))
    labels = np.array(label)[:,1]
    kf = KFold(data.shape[0], n_folds=5)
    scores = []
    for train_index, test_index in kf:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        batch_train, batch_test = batch[train_index], batch[test_index]
        K = np.dot(batch_train, np.transpose(batch_train))
        Kv = np.dot(batch_test, np.transpose(batch_train))
        clf.setK(K)
        clf.setKv(Kv)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))

    print np.mean(scores)

if __name__ == '__main__':
    run()