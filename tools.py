# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:40:46 2015

@author: Ying
"""
from sklearn import cross_validation
import pandas as pd

def load_data(url):
    data = pd.read_csv(url)
    return data

def split_data(data,cols):
    X=data[cols]
    y=data['Cover_Type']
    return X,y
    

def cv_score(clf,x_train,y_train):
    score=cross_validation.cross_val_score(clf, x_train, y_train, scoring=None, 
                                           cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')                                      
    return score.mean()