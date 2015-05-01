# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:39:22 2015

@author: Ying
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from tools import cv_score
from tools import load_data
import numpy as np
def fill_missing(data):
    x_train,y_train,x_test=split_missing(data)
    rfg=create_rfg()
    rfg.fit(x_train,y_train)
    data.Hillshade_3pm.loc[data.Hillshade_3pm==0]=np.around(rfg.predict(x_test))
    return data
def fill_missing_test(data):
    x_train,y_train,x_test=split_missing(data)
    rfg=create_rfg()
    rfg.fit(x_train[0:20000],y_train[0:20000])
    data.Hillshade_3pm.loc[data.Hillshade_3pm==0]=np.around(rfg.predict(x_test))
    return data
    
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
def cross_validation(data):
    X_train,y_train,X_test=split_missing(data)
    print cv_score(create_rfg(),X_train,y_train)
def combine(data):
    data['Soil']=0
    for i in range(1,41):
        data['Soil']=data['Soil']+i*data['Soil_Type'+str(i)]
    
    data['Wilderness_Area']=0
    for i in range(1,5):
         data['Wilderness_Area']=data['Wilderness_Area']+i*data['Wilderness_Area'+str(i)]
    return data
    
def preprocess_data(data):
    data=fill_missing(data)
    #data=combine(data)
    return data
