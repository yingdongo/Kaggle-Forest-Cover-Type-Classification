# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:23:06 2015

@author: Ying
"""
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from feature_engineering import feature_engineering
from data_preprocess import preprocess_data
from feature_selection import get_features
from feature_selection import split_data
from tools import load_data
from sklearn.grid_search import ParameterGrid
from tools import cv_score
def get_clfs():
    return {
            'extra trees' : { 
                'est' :ensemble.ExtraTreesClassifier(),
                'grid' : {
                    'n_estimators' : [400,500,600],
                    'max_features': ['sqrt', 'log2', .1, .3, None ],
                }
            },
#0.821825396825
#{'max_features': 0.3, 'n_estimators': 500}

            'random forests' : { 
                'est' :ensemble.RandomForestClassifier(),
                'grid' : {
                    'n_estimators' : [400,500,600],
                    'max_features': ['sqrt', 'log2', .1, .3, None ],
                }
            },
        }
#[0.81183862433862441, 'random forests', {'max_features': 'log2', 'n_estimators': 400}]

def grid_search(X_train,y,clfs):
    print "grid searching"
    for name,clf in clfs.iteritems(): 
            print name 
            param_grid=clfs[name]['grid']
            param_list = list(ParameterGrid(param_grid))
            for i in range(0,len(param_list)):
                   reg=clfs[name]['est'].set_params(**param_list[i])
                   cv=cv_score(reg,X_train,y)
                   print [cv.mean(),name,param_list[i]]

def grid_search1(X_train,y,clfs):
    for name, clf in clfs.iteritems(): 
        clf = GridSearchCV(clfs[name]['est'], clfs[name]['grid'], n_jobs=16, verbose=1, cv=10)
        clf.fit(X_train,y)
        print clf.score
        print clf.best_score_
        print clf.best_params_

def main():
    train=load_data('train.csv')
    preprocess_data(train)
    feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train,y_train=split_data(train,feature_cols)
    feature_cols=get_features(X_train,y_train)
    grid_search(X_train[feature_cols],y_train,get_clfs())

if __name__ == '__main__':
    main()