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
from matplotlib import pyplot as plt
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
def split_data(data,cols):
    X=data[cols]
    y=data['Cover_Type']
    return X,y
    
def create_clf():
    forest=ensemble.RandomForestClassifier()
    return forest
    
def feature_importances(X_train,y_train):
    clf=create_clf()
    clf.fit(X_train,y_train)
    return clf.feature_importances_
  
def cv_score(clf,X_train,y_train):
    score=cross_validation.cross_val_score(clf, X_train, y_train, scoring=None, 
                                           cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')
    return score.mean()

                                
def select_feature(X,y):
    estimator = SVC(kernel='linear')
    selector = RFECV(estimator, step=1, cv=5)
    selector = selector.fit(X, y)
    return selector


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
    train=preprocess_data('train.csv')
    add_feature(train)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train,y_train=split_data(train,feature_cols)
    selector=select_feature(X_train,y_train)
    print selector.support_ 
    print selector.ranking_


if __name__ == '__main__':
    main()
