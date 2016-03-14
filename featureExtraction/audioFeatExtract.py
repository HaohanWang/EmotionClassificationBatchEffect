
import sys
import glob
from math import floor, ceil
import numpy as np

class AudioFeat(object):

    def __init__(self):
        self.acousticT = 0.01

    def gen_acoustic_dict(self,featdir):
        acoustic_flst = glob.glob(featdir+'*.feat')
        self.acoustic_feat_dic = {}
        for fn in acoustic_flst:
            flst = []
            videoIdx = fn.split('/')[-1].split('(')[0][5:]
            for l in file(fn):
                lst = [float(f) for f in l.strip().split(',')]
                flst.append(lst)
            acoustic_feat = np.array(flst)
            self.acoustic_feat_dic[int(videoIdx)] = acoustic_feat

    def utt_audio_feature(self,videoIdx,start,end):
        a_start = floor(start/self.acousticT)
        a_end = ceil(end/self.acousticT)
        acoustic_feat = self.acoustic_feat_dic[videoIdx][a_start:a_end]
        
        vad_feat = acoustic_feat[:,0]
        pause = np.average(vad_feat)
        freq_feat = acoustic_feat[:,3]
        pitch = np.std(freq_feat)

        return [pause, pitch]
        
    
def read_utterance_time_label(fn):
    f = file(fn,'r')
    f.readline()
    utt_info_lst = []
    for line in f:
        lst = line.strip().split(',')
        videoIdx = int(lst[0])
        start = float(lst[1])
        if start<0: start=0
        end = float(lst[2])
        lab = int(lst[-1])
        utt_info_lst.append([videoIdx,start,end,lab])
    return utt_info_lst
    

if __name__=='__main__':
    audioFeat = AudioFeat()
    audioFeat.gen_acoustic_dict('../../features/AcousticFeatures/')

    featlst = []
    uttinfolst = read_utterance_time_label('../../annotations/sentiment/sentimentAnnotations1.csv')

    for uttinfo in uttinfolst:
        vidx, s, e, lab = uttinfo[0], uttinfo[1], uttinfo[2], uttinfo[3]
        feat = audioFeat.utt_audio_feature(vidx,s,e)
        featlst.append(feat)

    np.savetxt("../../data/audio_feature.csv", np.array(featlst), delimiter=",")

