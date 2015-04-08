# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:39:22 2015

@author: Ying
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
def fill_missing(data):
    x_train,y_train,x_test=split_missing(data)
    rfg=create_rfg()
    rfg.fit(x_train,y_train)
    data.Hillshade_3pm.loc[data.Hillshade_3pm==0]=rfg.predict(x_test)

def split_missing(data):    
    feature_cols_missing= [col for col in data.columns if col  not in ['Hillshade_3pm','Id']]
    X_train=data[feature_cols_missing][data.Hillshade_3pm!=0]
    y_train=data['Hillshade_3pm'][data.Hillshade_3pm!=0]
    X_test=data[feature_cols_missing][data.Hillshade_3pm==0]
    return X_train,y_train,X_test
    
def create_rfg():
    rforest=RandomForestRegressor(n_estimators=100)
    return rforest
def create_gbr():
    gbr=GradientBoostingRegressor(n_estimators=100)
    return gbr


def preprocess_data(data):
    fill_missing(data)
    return data
