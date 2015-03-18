# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 15:23:06 2015

@author: Ying
"""
from sklearn.grid_search import GridSearchCV
from sklearn import ensemble
from feature_engineering import add_feature
from explore_data import preprocess_data
from feature_selection import get_features
from feature_selection import split_data
def get_clfs():
    return {
            'extra trees' : { 
                'est' :ensemble.ExtraTreesClassifier(),
                'grid' : {
                    'n_estimators' : [100,500,1000],
                    'max_features': ['sqrt', 'log2', .1, .3, None ],
                }
            },
#0.810515873016
#{'max_features': 0.1, 'n_estimators': 500}
            'random forests' : { 
                'est' :ensemble.RandomForestClassifier(),
                'grid' : {
                    'n_estimators' : [100,500,1000],
                    'max_features': ['sqrt', 'log2', .1, .3, None ],
                }
            },
        }

#0.820899470899
#{'max_features': 0.3, 'n_estimators': 1000}
def grid_search(X_train,y,clfs):
    for name, clf in clfs.iteritems(): 
        clf = GridSearchCV(clfs[name]['est'], clfs[name]['grid'], n_jobs=16, verbose=0, cv=10)
        clf.fit(X_train,y)
        print clf.score
        print clf.best_score_
        print clf.best_params_

def main():
    train=preprocess_data('train.csv')
    add_feature(train)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train,y_train=split_data(train,feature_cols)
    feature_cols=get_features(X_train,y_train)
    grid_search(X_train[feature_cols],y_train,get_clfs())

if __name__ == '__main__':
    main()