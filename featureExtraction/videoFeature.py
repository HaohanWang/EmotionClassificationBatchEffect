__author__ = 'Haohan Wang'

import numpy as np
import csv

from label_utterance import get_video_length_label


def load_text(file_name):
    text = [line.strip() for line in open(file_name)]
    m = []
    for line in text:
        if len(line.split(',')) != 20:
            k = np.array([float(t) for t in line.split()])
        else:
            k = np.ones(134) * -1
        m.append(k)
    return np.array(m)


def main(featureLength=1, feature_extraction_methods=(None,), head_file=['tmp'], normalize=False):
    vll = get_video_length_label()
    result = [head_file]
    values = []
    for (v, s, e, label) in vll:
        fid = str(v)
        if len(fid) == 1:
            fid = '0' + fid
        filename = '../data/features/VisualFeatures/video' + fid + '_okao_output.txt'
        info = load_text(filename)
        feas = np.empty([featureLength])
        tmpIndex = 0
        for fem in feature_extraction_methods:
            l, r = fem(s, e, info)
            feas[tmpIndex:l + tmpIndex] = r
            tmpIndex = tmpIndex + l
        values.append(feas)
    values = np.array(values)
    if normalize:
        maxi = np.max(values, axis=0)
        mini = np.min(values, axis=0)
        values = (values - mini) / (maxi - mini)
    result.extend(values)
    labels = [['videoID', 'label']]
    labels.extend([[v, l] for (v, s, e, l) in vll])
    write_csv('../data/labels.csv', labels)
    write_csv('../data/video_features.csv', result)


def write_csv(filename, r):
    f = open(filename, 'w')
    w = csv.writer(f)
    w.writerows(r)
    f.close()


def getIndex(s, e, l):
    start = int(s * l)
    end = int(e * l) + 1
    return start, end


def getFeatureTitle():
    return ['smile50', 'smile75', 'gaze', 'head_vertical_mean', 'head_horizontal_mean',
            'head_roll_mean', 'head_v_std', 'head_h_std', 'head_r_std', 'head_v_std_count', 'head_h_std_count',
            'head_r_std_count', 'head_v_grad_mean', 'head_h_grad_mean', 'head_r_grad_mean',
            'head_v_grad_std', 'head_h_grad_std', 'head_r_grad_std', 'head_v_grad_std_count', 'head_h_grad_std_count',
            'head_r_std_count', 'head_size_mean', 'head_size_std', 'head_h_move_freq50', 'head_h_move_freq75',
            'head_v_move_freq50', 'head_v_move_freq75', 'head_r_move_freq50', 'head_r_move_freq75']


def std_counter(data, mean, std, count=1.0):
    t = mean.shape[0]
    diff = np.abs(data - mean)
    ind = np.where(diff >= std * count)[1]
    result = np.zeros(t)
    for i in range(t):
        result[i] = len(np.where(ind == i)[0])
    return result


def smile_featureExtraction(s, e, info, args=[50, 75]):
    """
    features = ['smile50', 'smile75']
    :param s:
    :param e:
    :param info:
    :param args: smile thresholds
    :return:
    """
    thresholds = args
    s, e = getIndex(s, e, info.shape[0])
    m = info[s:e, -1]
    m = m[np.where(m != -1)]
    l = float(len(m))
    # c1 = len(np.where(m <= thresholds[0])[0])
    c2 = len(np.where((m > thresholds[0]) & (m < thresholds[1]))[0]) / l
    c3 = len(np.where(m > thresholds[1])[0]) / l
    # mean = np.mean(m)
    # maxi = np.max(m)
    return 2, np.array([c2, c3])


def gaze_featureExtraction(s, e, info, args=[10]):
    """
    features = ['gaze']
    :param s:
    :param e:
    :param info:
    :param args: gaze focus (horizontal and vertical are the same)
    :return:
    """
    threshold = args[0]
    s, e = getIndex(s, e, info.shape[0])
    m = np.abs(info[s:e, -10:-7])
    m = m[np.where(m[:, 0] != -1)]
    l = float(len(m))
    c = len(np.where((m[:, 0] <= threshold) & (m[:, 1] <= threshold))[0])
    return 1, c / l


def posture_changes_featureExtraction(s, e, info, args=[]):
    """
    features = ['head_size_mean', 'head_size_std']
    :param s:
    :param e:
    :param info:
    :param args:
    :return:
    """
    s, e = getIndex(s, e, info.shape[0])
    m = info[:, :4]
    # mean = np.mean(m, axis=0)
    maxi = np.max(m, axis=0)
    mini = np.min(m, axis=0)
    m = m[s:e, :]
    m = m[np.where(m[:, 0] != -1)]
    # m -= mean
    m = (m-mini)/(maxi-mini)
    a = np.abs(m[:, 0] - m[:, 2])
    b = np.abs(m[:, 1] - m[:, 3])
    c = a * b
    t1 = c[:-1]
    t2 = c[1:]
    r1 = np.mean(t1)*np.mean(maxi-mini)
    r2 = np.std(t2)*np.mean(maxi-mini)
    return 2, np.array([r1, r2])


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.empty_like(y)
    for i in range(y.shape[1]):
        y_smooth[:, i] = np.convolve(y[:, i], box, mode='same')
    return y_smooth


def head_movement_featureExtraction(s, e, info, args=[]):
    """
    features = ['head_vertical_mean', 'head_horizontal_mean', 'head_roll_mean', 'head_v_std', 'head_h_std', 'head_r_std'
    'head_v_std_count', 'head_h_std_count', 'head_r_std_count', 'head_v_grad_mean', 'head_h_grad_mean', 'head_r_grad_mean'
    'head_v_grad_std', 'head_h_grad_std', 'head_r_grad_std', 'head_v_grad_std_count', 'head_h_grad_std_count', 'head_r_std_count']
    :param s:
    :param e:
    :param info:
    :param args:
    :return:
    """
    s, e = getIndex(s, e, info.shape[0])
    m = info[s:e, 120:123]
    m = m[np.where(m[:, 0] != -1)]
    mean = np.mean(info[:, 120:123], axis=0)
    m -= mean
    m1 = np.mean(m, axis=0)
    m2 = np.std(m, axis=0)
    m3 = std_counter(m, m1, m2, count=2)  # best count = 2
    t = np.gradient(smooth(m, 3))[0]
    t = np.gradient(t)[0]
    m4 = np.mean(t, axis=0)
    m5 = np.std(t, axis=0)
    m6 = std_counter(t, m4, m5, count=1.5)
    tmp = np.append(m1, m2, axis=1)
    tmp = np.append(tmp, m3, axis=1)
    tmp = np.append(tmp, m4, axis=1)
    tmp = np.append(tmp, m5, axis=1)
    return 18, np.append(tmp, m6, axis=1)


def head_movement_frequency_featureExtraction(s, e, info, args=[1.5, 10]):
    """
    features: 'head_v_freq', 'head_h_freq', 'head_r_freq'
    :param s:
    :param e:
    :param info:
    :param args:
    :return:
    """
    thresholds = args
    s, e = getIndex(s, e, info.shape[0])
    m = info[s:e, 120:123]
    m = m[np.where(m[:, 0] != -1)]
    m = np.abs(np.gradient(m)[1])
    # mean = np.mean(info[:, 120:123], axis=0)
    # m -= mean
    r = []
    for i in range(3):
        c1 = len(np.where((m[:, i] > thresholds[0]) & (m[:, i] < thresholds[1]))[0])
        c2 = len(np.where(m[:, i] > thresholds[1])[0])
        if c2 <= 3:
            c2 = 0
        r.append(c1)
        r.append(c2)
    return 6, np.array(r)


if __name__ == '__main__':
    title = getFeatureTitle()
    main(featureLength=len(title),
         feature_extraction_methods=(smile_featureExtraction, gaze_featureExtraction, head_movement_featureExtraction,
                                     posture_changes_featureExtraction, head_movement_frequency_featureExtraction),
         head_file=title)
