# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:55:24 2015

@author: Ying
"""
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble

from feature_engineering import cv_score

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
    score=[]    
    for clf in models:
        score.append((clf[0],cv_score(clf[1],)))
    return score