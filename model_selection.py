# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:55:24 2015

@author: Ying
"""
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from feature_engineering import feature_engineering
from tools import load_data
from data_preprocess import preprocess_data
from feature_selection import cv_score
from feature_selection import split_data
import pandas as pd
from matplotlib import pyplot as plt
from feature_selection import get_features

def create_clf():
    models=[]
    models.append(('linearSVC',LinearSVC()))
    models.append(('KNN',KNeighborsClassifier()))
    models.append(('AdaBooost',ensemble.AdaBoostClassifier()))
    models.append(('Bagging',ensemble.BaggingClassifier()))
    models.append(('ExtraTrees',ensemble.ExtraTreesClassifier()))
    models.append(('GB',ensemble.GradientBoostingClassifier()))
    models.append(('RandomForest',ensemble.RandomForestClassifier()))
    return models

def clf_score(models,X_train,y_train):
    index=[]
    score=[]
    for clf in models:
        index.append(clf[0])
        score.append(cv_score(clf[1],X_train,y_train))
    return pd.DataFrame(score,index=index)
def main():
    train=load_data('train.csv')
    preprocess_data(train)
    feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train,y_train=split_data(train,feature_cols)
    #feature_cols=get_features(X_train,y_train)
    clf_scores=clf_score(create_clf(),X_train[feature_cols],y_train)
    print clf_scores
    plt.plot(clf_scores)
    plt.xticks(range(len(clf_scores)), clf_scores.index, fontsize=14, rotation=90)
    plt.show()

if __name__ == '__main__':
    main()
#60features
#linearSVC     0.323082
#KNN           0.660780
#AdaBooost     0.445304
#Bagging       0.765013
#ExtraTrees    0.794246
#GB            0.739484
#RandomForest  0.781415

#total features
#linearSVC     0.326587
#KNN           0.660780
#AdaBooost     0.445304
#Bagging       0.771429
#ExtraTrees    0.791204
#GB            0.740146
#RandomForest  0.776124