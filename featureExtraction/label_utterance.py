__author__ = 'Haohan Wang'

def get_video_length_label():
    '''
    get the # of video, its start and end in percentage of the video, its label (0 neutral, 1 positive, 2 negative)
    :return:
    '''
    text = [line.strip() for line in open('../data/sentimentAnnotations.csv')][1:]
    r = []
    d = {}
    for line in text:
        items = line.split(',')
        v = int(items[0])
        s = float(items[1])
        if s < 0:
            s = 0
        e = float(items[2])
        label = int(items[-1])
        if label == -1:
            label = 2
        r.append([v, s, e, label])
        d[v] = e
    for m in r:
        m[1] = m[1]/d[m[0]]
        m[2] = m[2]/d[m[0]]
    return r

def label_test():
    from scipy.stats import mode
    text = [line.strip() for line in open('../data/sentimentAnnotations.csv')][1:]
    r = []
    d = {}
    for line in text:
        labels = [0, 0, 0]
        items = line.split(',')
        labels[0] = int(items[-4])
        labels[1] = int(items[-3])
        labels[2] = int(items[-2])
        r = mode(labels)
        # print r,
        if r[1] == 1:
            print 0
        else:
            print r[0][0]
        # print l
    return r

if __name__ == '__main__':
    label_test()