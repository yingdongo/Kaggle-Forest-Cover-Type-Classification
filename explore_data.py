# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:23:41 2015

@author: Ying
"""

#import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation
import ggplot as gp

def load_data(url):
    data = pd.read_csv(url)
    return data

picsPath = ""

def plotHist(data, x,c):
    
    p = gp.ggplot(gp.aes(x=x,fill=c), data=data)
    p = p + gp.geom_histogram()
    p = p + gp.ggtitle("Histogram - %s" % (str(x)))
    gp.ggsave(p, "%stemp/Histogram-%s.png" % (picsPath, str(x)))


def plotPoints(data, x, y,c):
    p = gp.ggplot(gp.aes(x=x, y=y,colour=c), data=data)
    p = p + gp.geom_point()
    p = p + gp.ggtitle("Scatter - %s vs. %s" % (str(x), str(y)))
    gp.ggsave(p, "%stemp/Scatter-%s_vs_%s.png" % (picsPath, str(x), str(y)))

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

def cv_score(clf,x_train,y_train):
    score=cross_validation.cross_val_score(clf, x_train, y_train, scoring=None, 
                                           cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')                                      
    return score.mean()
    
def fill_missing(data):
    x_train,y_train,x_test=split_missing(data)
    rfg=create_rfg()
    rfg.fit(x_train,y_train)
    data.Hillshade_3pm.loc[data.Hillshade_3pm==0]=rfg.predict(x_test)
    
def preprocess_data(url):
    data=load_data(url)
    fill_missing(data)
    return data
train = load_data('train.csv')
fill_missing(train)    

#def main():   
#==============================================================================
#    cols = [col for col in train.columns if col not in train.columns[11:56] and  col not in ['Id']] 
#    for col in cols:
#        plotHist(train, col,xCol)
#==============================================================================

#==============================================================================
#     xCol = 'Cover_Type'
#     yCols = [col for col in train.columns if col not in train.columns[11:56] and col not in ['Id', 'Cover_Type']] 
#     for i in range(len(yCols)):
#         for j in range(len(yCols)):
#             if i!=len(yCols)-1-j:
#                 plotPoints(train, yCols[i], yCols[len(yCols)-1-j],xCol)
#==============================================================================

#if __name__ == '__main__':
#    main()

