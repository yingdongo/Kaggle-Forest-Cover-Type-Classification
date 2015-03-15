# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 22:55:01 2015

@author: Ying
"""
from sklearn import ensemble
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation



def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train,test

def create_rfg():
    rforest=RandomForestRegressor(n_estimators=100)
    return rforest
    
def fill_missing(data):
    x_train,y_train,x_test=split_missing(data)
    rfg=create_rfg()
    rfg.fit(x_train,y_train)
    data.Hillshade_3pm.loc[data.Hillshade_3pm==0]=rfg.predict(x_test)
    
def split_data(data,cols):
    X_train=data[cols]
    y_train=data['Cover_Type']
    return X_train,y_train
    
def split_missing(data):    
    feature_cols_missing= [col for col in data.columns if col  not in ['Hillshade_3pm','Id']]
    X_train=data[feature_cols_missing][data.Hillshade_3pm!=0]
    y_train=data['Hillshade_3pm'][data.Hillshade_3pm!=0]
    X_test=data[feature_cols_missing][data.Hillshade_3pm==0]
    return X_train,y_train,X_test
def create_clf():
    forest=ensemble.ExtraTreesClassifier(n_estimators=400, criterion='gini', max_depth=None,max_features=None)
    return forest
    
def feature_importances(X_train,y_train):
    clf=create_clf()
    clf.fit(X_train,y_train)
    return pd.DataFrame(clf.feature_importances_,index=X_train.columns).sort([0], ascending=False)
  
def cv_score(clf,X_train,y_train):
    score=cross_validation.cross_val_score(clf, X_train, y_train, scoring=None, 
                                           cv=10, n_jobs=1, verbose=0, fit_params=None, score_func=None, 
                                           pre_dispatch='2*n_jobs')
    return score.mean()
                                
def select_feature(data,feature_cols):
    score=pd.DataFrame()
    for i in range(5,60):
        cols=feature_cols[:i]
        X_train,y_train=split_data(data,cols)
        score.append([cv_score(create_clf(),X_train,y_train),i])
        
train,test=load_data()

def r(x):
    if x+180>360:
        return x-180
    else:
        return x+180
def add_feature():
    train['Aspect2']=train.Aspect.map(r)
    test['Aspect2']=test.Aspect.map(r)
    
    train['Ele_minus_VDtHyd'] = train.Elevation-train.Vertical_Distance_To_Hydrology
    test['Ele_minus_VDtHyd'] = test.Elevation-test.Vertical_Distance_To_Hydrology
         
    train['Ele_plus_VDtHyd'] = train.Elevation+train.Vertical_Distance_To_Hydrology
    test['Ele_plus_VDtHyd'] = test.Elevation+test.Vertical_Distance_To_Hydrology
     
    train['Distanse_to_Hydrolody'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
    test['Distanse_to_Hydrolody'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
     
    train['Hydro_plus_Fire'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
    test['Hydro_plus_Fire_plus'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
     
    train['Hydro_minus_Fire'] = train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points']
    test['Hydro_minus_Fire'] = test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points']
     
    train['Hydro_plus_Road'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways']
    test['Hydro_plus_Road'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways']
     
    train['Hydro_minus_Road'] = train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways']
    test['Hydro_minus_Road'] = test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways']
     
    train['Fire_plus_Road'] = train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways']
    test['Fire_plus_Road'] = test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways']
     
    train['Fire_minus_Road'] = train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways']
    test['Fire_minus_Road'] = test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways']
    
    
    train['Soil']=0
    for i in range(1,41):
        train['Soil']=train['Soil']+i*train['Soil_Type'+str(i)]
     
    test['Soil']=0
    for i in range(1,41):
        test['Soil']=test['Soil']+i*test['Soil_Type'+str(i)]
     
    train['Wilderness_Area']=0
    for i in range(1,5):
         train['Wilderness_Area']=train['Wilderness_Area']+i*train['Wilderness_Area'+str(i)]
      
    test['Wilderness_Area']=0
    for i in range(1,5):
         test['Wilderness_Area']=test['Wilderness_Area']+i*test['Wilderness_Area'+str(i)]

feature_cols= [col for col in train.columns if col  not in ['Cover_type','Id']]

X_train,y_train=split_data(train,feature_cols)

f=feature_importances(X_train,y_train)