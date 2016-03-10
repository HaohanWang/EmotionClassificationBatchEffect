__author__ = 'Haohan Wang'

import numpy as np

def load_data_label(FeatureFile, LabelFile, mapping=None):
    if mapping is None:
        # title = getFeatureTitle()
        # mapping = np.arange(len(title))
        mapping = [21, 22]
        # all: 0.3928, 0.2216
        # origin: 0.3857, 0.3425
        # best 3, 6, 9: 0.4392 (F1 0.4341) For label 1: 0.28, For label 0, 0.22
        # 21, 22: 0.4321 (F1 0.4515) For label 2: 0.54 # probably not
        # 27, 28: 0.4142 (F1 0.4418) For label 2: 0.5287
    data = np.loadtxt(FeatureFile, skiprows=1, delimiter=',')
    labels = np.loadtxt(LabelFile, skiprows=1, delimiter=',')
    m = np.zeros(data.shape[1])
    for k in mapping: m[k] = 1
    ind = np.where(m == 1)
    data = data[:, ind[0]]
    return data, labels.astype(int)


if __name__ == '__main__':
    data, label = load_data_label('../data/video_features.csv', '../data/labels.csv')
    print len(label)
    print data.shape
