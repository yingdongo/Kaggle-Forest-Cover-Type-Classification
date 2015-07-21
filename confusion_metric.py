# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:47:10 2015

@author: Ying
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:55:49 2015

@author: Ying
"""
from sklearn import metrics
import numpy as np, pylab as pl
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from feature_engineering import feature_engineering
from data_preprocess import preprocess_data

def sep_clf(clf,train,X_test,feature_cols):   
    seq=np.array([4,7,5,6,3])
    #pres=np.ndarray(shape=(7,len(test)))
    pres=[]
    lens=[]
    ids=np.array(np.arange(0, len(X_test)))

    for i in seq:
        train_num=train.copy()
        train_num['Cover_Type'][train_num['Cover_Type']<>i]=0
        X_train = train_num[feature_cols]
        y = train_num['Cover_Type']
        clf.fit(X_train,y)
        pre=clf.predict(X_test)
        id_a=np.where(pre==i)
        id_i=ids[pre==i]
        pres=np.append(pres,id_i)
        lens=np.append(lens,len(id_i))
        X_test=X_test.drop(X_test.index[id_a])
        ids=ids[pre<>i]
    
    train_num=train.copy()
    train_num=train_num[train_num['Cover_Type']<3]
    X_train = train_num[feature_cols]  
    y = train_num['Cover_Type']
    clf.fit(X_train,y)
    pre=clf.predict(X_test)
    
   # id_i=ids[pre==3]
   # pres=np.append(pres,id_i)
   # lens=np.append(lens,len(id_i))
    
    id_i=ids[pre==2]
    pres=np.append(pres,id_i)
    lens=np.append(lens,len(id_i))
    
    id_i=ids[pre==1]
    pres=np.append(pres,id_i)
    lens=np.append(lens,len(id_i))
    
    lens=lens.astype(int)
    pres=pres.astype(int)
    ctype=np.array(np.zeros(len(pres)),dtype=int)
    running_sum=np.cumsum(lens)
    
    ctype[0:running_sum[0]]=4
    ctype[running_sum[0]-2:running_sum[1]]=7
    ctype[running_sum[1]-2:running_sum[2]]=5
    ctype[running_sum[2]-2:running_sum[3]]=6
    ctype[running_sum[3]-2:running_sum[4]]=3
    ctype[running_sum[4]-2:running_sum[5]]=2
    ctype[running_sum[5]-2:running_sum[6]]=1
    p=pres.argsort()
    y_pred=ctype[p]
    return y_pred
    
train = pd.read_csv('train.csv')
train=preprocess_data(train)
train=feature_engineering(train)

feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id','Vertical_Distance_To_Hydrology','Slope','Soil_Type7','Soil_Type8','Soil_Type15']]

X= train[feature_cols]
y = train['Cover_Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,random_state=42)

forest=ensemble.ExtraTreesClassifier(max_features= 0.35, n_estimators= 600)
                                     
forest.fit(X_train,y_train)
pred=forest.predict(X_test)

categories=['1','2','3','4','5','6','7']

# get overall accuracy and F1 score to print at top of plot
pscore = metrics.accuracy_score(y_test, pred)
score = metrics.f1_score(y_test, pred, pos_label=list(set(y_test)))
# get size of the full label set
dur = len(categories)
print "Building testing confusion matrix..."
# initialize score matrices
trueScores = np.zeros(shape=(dur,dur))
predScores = np.zeros(shape=(dur,dur))
# populate totals
for i in xrange(len(y_test)-1):
  trueIdx = y_test.values[i]-1
  predIdx = pred[i]-1
  trueScores[trueIdx,trueIdx] += 1
  predScores[trueIdx,predIdx] += 1
# create %-based results
trueSums = np.sum(trueScores,axis=0)
conf = np.zeros(shape=predScores.shape)
for i in xrange(len(predScores)):
  for j in xrange(dur):
    conf[i,j] = predScores[i,j] / trueSums[i]
# plot the confusion matrix
hq = pl.figure(figsize=(12,12));
aq = hq.add_subplot(1,1,1)
aq.set_aspect(1)
res = aq.imshow(conf,cmap=pl.get_cmap('Greens'),interpolation='nearest',vmin=-0.05,vmax=1.)
width = len(conf)
height = len(conf[0])
done = []
# label each grid cell with the misclassification rates
for w in xrange(width):
  for h in xrange(height):
      pval = conf[w][h]
      c = 'k'
      rais = w
      if pval > 0.5: c = 'w'
      if pval > 0.001:
        if w == h:
          aq.annotate("{0:1.1f}%\n{1:1.0f}/{2:1.0f}".format(pval*100.,predScores[w][h],trueSums[w]), xy=(h, w), 
                  horizontalalignment='center',
                  verticalalignment='center',color=c,size=16)
        else:
          aq.annotate("{0:1.1f}%\n{1:1.0f}".format(pval*100.,predScores[w][h]), xy=(h, w), 
                  horizontalalignment='center',
                  verticalalignment='center',color=c,size=16)
# label the axes
pl.xticks(range(width), categories[:width],rotation=90,size=16)
pl.yticks(range(height), categories[:height],size=16)
# add a title with the F1 score and accuracy
aq.set_title(" Prediction, Test Set (f1: "+"{0:1.3f}".format(score)+', accuracy: '+'{0:2.1f}%'.format(100*pscore)+", " + str(len(y_test)) + " items)",fontname='Arial',size=10,color='k')
aq.set_ylabel("Actual",fontname='Arial',size=16,color='k')
aq.set_xlabel("Predicted",fontname='Arial',size=16,color='k')
pl.grid(b=True,axis='both')
# save it
pl.savefig("confusion.png")

