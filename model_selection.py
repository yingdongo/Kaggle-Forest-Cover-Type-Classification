# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:55:24 2015

@author: Ying
"""
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from feature_engineering import add_feature
from explore_data import preprocess_data
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
    train=preprocess_data('train.csv')
    add_feature(train)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train,y_train=split_data(train,feature_cols)
    cols=get_features(X_train,y_train)
    clf_scores=clf_score(create_clf(),X_train[cols],y_train)
    print clf_scores
    plt.plot(clf_scores)
    plt.xticks(range(len(clf_scores)), clf_scores.index, fontsize=14, rotation=90)
    plt.show()

if __name__ == '__main__':
    main()
