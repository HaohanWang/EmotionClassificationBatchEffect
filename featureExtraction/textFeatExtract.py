import sys
import glob
import nltk
import numpy as np
from nltk.util import ngrams
from nltk.corpus import stopwords

class TextFeat(object):

    def __init__(self):
        self.unigramSet = None
        self.rawText = None
        self.stop_word_set = nltk.corpus.stopwords.words('english')
        self.stemmer = nltk.stem.snowball.SnowballStemmer('english')

    def unigram_accu(self,textfn):
        if self.unigramSet is None: self.unigramSet = set()
        for starttime, txt in self.yield_text(textfn):
            tokens = nltk.word_tokenize(txt)
            tokens = [self.stemmer.stem(token) for token in tokens]
            tokens = [token for token in tokens if token not in self.stop_word_set]
            unigram = ngrams(tokens,1)
            for gram in unigram: self.unigramSet.add(gram)

    def yield_text(self,textfn):
        txtfile = file(textfn)
        while True:
            line = txtfile.readline()
            if len(line)==0: break
            if line[:3]=='<Sy':
                starttime = float(synstr_to_time(line))
                txt = txtfile.readline().strip().lower()
                txt = txt.replace(':','')
                yield starttime, txt
    
    def calculate_word_freq(self,textfnlst):
        tokens=[]
        for textfn in textfnlst:
            self.wholetext_accu(textfn)
        tokens = nltk.word_tokenize(self.rawText)
        #tokens = [self.stemmer.stem(token) for token in tokens]
        tokens = [token for token in tokens if token not in self.stop_word_set]
        self.fd = nltk.FreqDist(tokens)

    def wholetext_accu(self,textfn):
        if self.rawText is None: self.rawText = ''
        for starttime, txt in self.yield_text(textfn):
            self.rawText+=' '
            self.rawText+=txt
    
    def utt_text_feature(self,fn,start,end):
        alltxt = ''
        feat = np.array([0]*100)
        for starttime, txt in self.yield_text(fn):
            if starttime>start and starttime<end:
                alltxt+=(' '+txt)
        tokens = nltk.word_tokenize(alltxt)
        uniSet = self.fd.most_common(100)
        uniSet = [pr[0] for pr in uniSet]
        for token in tokens:
            if token in uniSet: 
                idx = uniSet.index(token)
                feat[idx]=1
        return feat
        
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

def synstr_to_time(syn):
    return syn.split('\"')[1]

def main(text_feat_dir):
    textfnlst = glob.glob(text_feat_dir+'/*.trs')
    textFeat = TextFeat()
    for textfn in textfnlst:
        textFeat.unigram_accu(textfn)
    textFeat.calculate_word_freq(textfnlst)

    featlst = []
    uttinfolst = read_utterance_time_label('../../annotations/sentiment/sentimentAnnotations1.csv')
    for uttinfo in uttinfolst:
        vidx, s, e, lab = uttinfo[0], uttinfo[1], uttinfo[2], uttinfo[3]
        feat = textFeat.utt_text_feature(text_feat_dir+'/video'+str(vidx)+'.trs',s,e)
        featlst.append(feat)

    np.savetxt("../../data/text_feature.csv", np.array(featlst), delimiter=",")
        
if __name__=='__main__':
    main('../../annotations/Transcriptions/')
