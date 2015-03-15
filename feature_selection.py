# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:55:01 2015

@author: Ying
"""
from sklearn import ensemble
import pandas as pd
from sklearn import cross_validation
from feature_engineering import add_feature
from explore_data import preprocess_data

import numpy as np

def split_data(data,cols):
    X_train=data[cols]
    y_train=data['Cover_Type']
    return X_train,y_train
    
def create_clf():
    forest=ensemble.ExtraTreesClassifier(n_estimators=400, criterion='gini', max_depth=None,max_features=None)
    return forest
    
def feature_importances(X_train,y_train):
    clf=create_clf()
    clf.fit(X_train,y_train)
    return pd.DataFrame(clf.feature_importances_,index=X_train.columns).sort([0], ascending=False)
  
def cv_score(clf,X_train,y_train):
    score=cross_validation.cross_val_score(clf, X_train, y_train, scoring=None, 
                                           cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')
    return score.mean()
                                
def select_feature(data,feature_cols):
    f_count=66
    f_start=np.int(np.sqrt(f_count))
    f_range=range(f_start,f_count)
    score=np.array(np.zeros(f_count)-f_start)
    for i in f_range:
        cols=feature_cols[:i]
        X_train,y_train=split_data(data,cols)
        score[i-f_start]=cv_score(create_clf(),X_train,y_train)
    return pd.DataFrame(score,index=f_range).sort([0], ascending=False)


train=preprocess_data('train.csv')
add_feature(train)
feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
X_train,y_train=split_data(train,feature_cols)

f=feature_importances(X_train,y_train)

score=select_feature(train,f.index)