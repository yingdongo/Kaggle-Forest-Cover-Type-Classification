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

def create_clf():
    forest=ensemble.RandomForestClassifier()
    return forest
    
def feature_importances(X_train,y_train):
    clf=create_clf()
    clf.fit(X_train,y_train)
    return clf.feature_importances_
  

                                
def select_feature(data,feature_cols,importances):
    indices = np.argsort(importances)[::-1]    
    f_count=64
    f_start=np.int(np.sqrt(f_count))
    f_range=range(f_start,f_count)
    score=np.array(np.zeros(f_count-f_start))
    for i in f_range:
        cols=feature_cols[indices]
        cols=cols[:i]
        X_train,y_train=split_data(data,cols)
        score[i-f_start]=cv_score(create_clf(),X_train,y_train)
    return pd.DataFrame(score,index=f_range)


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
    return cols[:54]

def main():
    train=load_data('train.csv')
    preprocess_data(train)
    feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train,y_train=split_data(train,feature_cols)
    importances=feature_importances(X_train,y_train)
    plot_importances(importances,X_train.columns)
    score=select_feature(train,X_train.columns,importances)
    print score
    plt.figure(figsize=(20,8))
    plt.plot(score)
    plt.xticks(range(len(score)),score.index)
    plt.show()


if __name__ == '__main__':
    main()
