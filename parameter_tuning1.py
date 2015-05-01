# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:23:06 2015

@author: Ying
"""
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from feature_engineering import feature_engineering
from data_preprocess import preprocess_data
from tools import load_data
from feature_selection import split_data
from sklearn.grid_search import ParameterGrid
from tools import cv_score

def get_clfs1():
    return {
            'extra trees' : { 
                'est' :ensemble.ExtraTreesClassifier(),
                'grid' : {
                    'n_estimators' : [500,550,600],
                    'max_features': [.3,.35,None],
                }
            },
#[0.82275132275132279, 'extra trees', {'max_features': 0.35, 'n_estimators': 600}]
#0.822288359788
#{'max_features': 0.35, 'n_estimators': 600}
#0.821825396825
#{'max_features': 0.3, 'n_estimators': 500}
#0.821957671958
#{'max_features': 0.3, 'n_estimators': 600}
#0.821891534392
#{'max_features': 0.35, 'n_estimators': 550}
#0.822552910053
#{'max_features': 0.35, 'n_estimators': 590}
             'random forests' : { 
                'est' :ensemble.RandomForestClassifier(),
                'grid' : {
                    'n_estimators' : [350,400,500],
                    'max_features': [0.6,'log2', None ],
                }
            },
        }
#[0.81335978835978828, 'random forests', {'max_features': 'log2', 'n_estimators': 400}]

#0.810714285714
#{'max_features': 'log2', 'n_estimators': 400}
def get_clfs():
    return {
            'extra trees' : { 
                'est' :ensemble.ExtraTreesClassifier(),
                'grid' : {
                    #'n_estimators' : [600,650,700],
                    #'max_features': [.35,.4,.5],
                    #'n_estimators' : [600,700,800],
                    'n_estimators' : [600,800,900],
                    'max_features': [.4,.5,.6,.7]
                }
            }
           }
#[0.82361111111111107, 'extra trees', {'max_features': 0.6, 'n_estimators': 600}]
#[0.82361111111111107, 'extra trees', {'max_features': 0.5, 'n_estimators': 800}]
#[0.82275132275132279, 'extra trees', {'max_features': 0.5, 'n_estimators': 600}]
#0.822288359788
#{'max_features': 0.35, 'n_estimators': 600}
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
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id','Vertical_Distance_To_Hydrology','Slope','Soil_Type7','Soil_Type8','Soil_Type15']]
    X_train,y_train=split_data(train,feature_cols)
    grid_search(X_train[feature_cols],y_train,get_clfs())

if __name__ == '__main__':
    main()