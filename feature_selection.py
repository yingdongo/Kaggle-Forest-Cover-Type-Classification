# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:55:01 2015

@author: Ying
"""
from sklearn import ensemble
import pandas as pd
from tools import load_data
from tools import cv_score
from tools import split_data
from feature_engineering import feature_engineering
from data_preprocess import preprocess_data
from matplotlib import pyplot as plt
import numpy as np
plt.style.use('ggplot')

def create_clf():
    forest=ensemble.RandomForestClassifier(n_estimators=100)
    return forest
    
def create_extra():
    forest=ensemble.ExtraTreesClassifier(n_estimators=100)
    return forest
    
def feature_importances(X_train,y_train):
    clf=create_clf()
    clf.fit(X_train,y_train)
    return clf.feature_importances_
  
def select_feature(data,feature_cols,importances):
    indices = np.argsort(importances)[::-1]    
    f_count=65
    f_start=3
    f_range=range(f_start,f_count)
    score=np.array(np.zeros(f_count-f_start))
    for i in f_range:
        cols=feature_cols[indices]
        cols=cols[:i]
        X_train,y_train=split_data(data,cols)
        score[i-f_start]=cv_score(create_clf(),X_train,y_train)
    return pd.DataFrame(score,index=f_range)
    
def get_score(data,cols):
    X_train,y_train=split_data(data,cols)
    return cv_score(create_extra(),X_train,y_train)
    

def plot_importances(importances, col_array):
# Calculate the feature ranking
    indices = np.argsort(importances)[::-1]    
#Mean Feature Importance
    print "\nMean Feature Importance %.6f" %np.mean(importances)    
#Plot the feature importances of the forest
    plt.figure(figsize=(20,8))
    plt.title("Feature importances")
    plt.bar(range(len(importances)), importances[indices],
            color="gr", align="center")
    plt.xticks(range(len(importances)), col_array[indices], fontsize=14, rotation=90)
    plt.xlim([-1, len(importances)])
    plt.show()
    
def get_features(X_train,y_train):
    importances=feature_importances(X_train,y_train)
    indices = np.argsort(importances)[::-1]   
    cols=X_train.columns[indices]
    return cols[:60]
    
def plot_correlations(data):
    """Plot pairwise correlations of features in the given dataset"""

    from matplotlib import cm
    
    cols = data.columns.tolist()
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    
    # Plot absolute value of pairwise correlations since we don't
    # particularly care about the direction of the relationship,
    # just the strength of it
    cax = ax.matshow(data.corr().abs(), cmap=cm.YlOrRd)
    
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols,rotation=90)
    ax.set_yticklabels(cols)
    
def main():
    train=load_data('train.csv')
    preprocess_data(train)
    feature_engineering(train)
    cols = [col for col in train.columns if col not in train.columns[11:56] and  col not in ['Id']] 
    for col in enumerate(list(cols)):
         feature_cols=cols    
         feature_cols.remove(col)
         get_score(train,feature_cols)
    #plot_correlations(train[cols])
    #feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    #X_train,y_train=split_data(train,feature_cols)
    #importances=feature_importances(X_train,y_train)
    #plot_importances(importances,X_train.columns)
    #score=select_feature(train,X_train.columns,importances)
    #print score
    #plt.figure(figsize=(20,8))
    #plt.plot(score)
    #plt.xticks(range(len(score)),score.index)
    #plt.show()


#if __name__ == '__main__':
#    main()

#43  0.810450 60  0.810119 24  0.799537
def select():
    train=load_data('train.csv')
    preprocess_data(train)
    feature_engineering(train)
    cols1 = [col for col in train.columns if col not in ['Id','Cover_Type','Soil_Type7','Soil_Type8','Soil_Type15']] 
    cols = [col for col in train.columns if col not in train.columns[11:56] and  col not in ['Id','Cover_Type']] 
    print 'total'
    print get_score(train,cols1)
    scores=np.array(np.zeros(19))
    for e,col in enumerate(list(cols)):      
          feature_cols=list(cols1)    
          feature_cols.remove(col)
          get_score(train,feature_cols)
          score=get_score(train,feature_cols)
          print col 
          print score
          scores[e]=score
    return pd.DataFrame(data=scores,index=cols)
