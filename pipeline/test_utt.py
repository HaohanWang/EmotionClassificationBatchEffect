import sys
from util_ml import *
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

featfn = 'video_features.csv'
labfn = 'labels.csv'
textfeatfn = 'text_feature.csv'
audiofeatfn = 'audio_feature.csv'

X_v = np.genfromtxt(featfn,dtype=float, delimiter=',', skip_header=1)
X_t = np.genfromtxt(textfeatfn,dtype=float, delimiter=',', skip_header=0)
X_a = np.genfromtxt(audiofeatfn,dtype=float, delimiter=',', skip_header=0)
lab = np.genfromtxt(labfn, delimiter=',', skip_header=1)

Xnorm_a = np.zeros_like(X_a)
Xnorm_v = np.zeros_like(X_v)
videoIdxs = lab[:,0]
videoSet = list(set(videoIdxs.tolist()))
y = lab[:,1]
acclst = []

for v in videoSet:
    Xnorm_v[np.where(videoIdxs==v)] = stats.zscore(X_v[np.where(videoIdxs==v)],axis=0)
    Xnorm_a[np.where(videoIdxs==v)] = stats.zscore(X_a[np.where(videoIdxs==v)],axis=0)
Xnorm_v = np.nan_to_num(Xnorm_v)
Xnorm_a = np.nan_to_num(Xnorm_a)


X = np.concatenate((Xnorm_v,X_t,Xnorm_a),axis=1)
#X = np.concatenate((Xnorm_v,X_t),axis=1)
#X = Xnorm_a
X = X[np.where(y>0)]
y = y[np.where(y>0)]
np.random.seed(1234)
X, y = RandomPerm(X,y)
ytest_total = []
ypred_total = []
for Xtrain, ytrain, Xtest, ytest in KFold(X,y,10):
    #clf = SVC(C=0.1)
    clf = SVC(C=10)
    #clf = SVC(C=1000)
    #clf = SVC(kernel='linear',C=100)
    #clf = DecisionTreeClassifier(max_depth=50)
    #clf = RandomForestClassifier(max_depth=5,n_estimators=10)
    clf.fit(Xtrain,ytrain)
    ypred = clf.predict(Xtest)
    print ypred
    ytest_total+=ytest.tolist()
    ypred_total+=ypred.tolist()

print f1_score(ytest_total,ypred_total)
print accuracy_score(ytest_total,ypred_total)
