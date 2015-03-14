# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:23:41 2015

@author: Ying
"""

#import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation

def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train,test

train, test=load_data()
feature_cols = [col for col in train.columns if col not in train.columns[11:55] and col  not in ['Cover_Type','Id']]

def histogram(data):  
    train[feature_cols].hist(figsize=(16,12),bins=50)
    plt.show()

#Looks like there're some missing values for Hillshade at 3 PM where value=0
def split_missing(data):    
    feature_cols_missing= [col for col in data.columns if col  not in ['Hillshade_3pm','Id']]
    x_train=data[feature_cols_missing][data.Hillshade_3pm!=0]
    y_train=data['Hillshade_3pm'][data.Hillshade_3pm!=0]
    x_test=data[feature_cols_missing][data.Hillshade_3pm==0]
    return x_train,y_train,x_test
    
def cv_missing_rf(x_train,y_train):
    rforest=RandomForestRegressor(n_estimators=100)
    score=cross_validation.cross_val_score(rforest, x_train, y_train, scoring=None, 
                                           cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')
    return score
def cv_gbr(x_train,y_train):
    gbr=GradientBoostingRegressor(n_estimators=100)
    score=cross_validation.cross_val_score(gbr, x_train, y_train, scoring=None, 
                                           cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')
    return score
x_train,y_train,x_test=split_missing(test)

cv_score=cv_missing_rf(x_train,y_train)