# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 17:25:46 2015

@author: Ying
"""

from sklearn import ensemble
import pandas as pd
from tools import load_data
from tools import cv_score
from tools import split_data
from data_preprocess import preprocess_data
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('ggplot')

def feature_engineering(data):
    
    data['Ele_minus_VDtHyd'] = data.Elevation-data.Vertical_Distance_To_Hydrology
         
    data['Ele_plus_VDtHyd'] = data.Elevation+data.Vertical_Distance_To_Hydrology
     
    data['Distanse_to_Hydrolody'] = (data['Horizontal_Distance_To_Hydrology']**2+data['Vertical_Distance_To_Hydrology']**2)**0.5
     
    data['Hydro_plus_Fire'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Fire_Points']
     
    data['Hydro_minus_Fire'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Fire_Points']
     
    data['Hydro_plus_Road'] = data['Horizontal_Distance_To_Hydrology']+data['Horizontal_Distance_To_Roadways']
     
    data['Hydro_minus_Road'] = data['Horizontal_Distance_To_Hydrology']-data['Horizontal_Distance_To_Roadways']
     
    data['Fire_plus_Road'] = data['Horizontal_Distance_To_Fire_Points']+data['Horizontal_Distance_To_Roadways']
     
    data['Fire_minus_Road'] = data['Horizontal_Distance_To_Fire_Points']-data['Horizontal_Distance_To_Roadways']
    
    data['Soil']=0
    for i in range(1,41):
        data['Soil']=data['Soil']+i*data['Soil_Type'+str(i)]
    
    data['Wilderness_Area']=0
    for i in range(1,5):
         data['Wilderness_Area']=data['Wilderness_Area']+i*data['Wilderness_Area'+str(i)]
    return data

def create_clf():
    forest=ensemble.RandomForestClassifier(n_estimators=100)
    return forest
    
def feature_importances(X_train,y_train):
    clf=create_clf()
    clf.fit(X_train,y_train)
    return clf.feature_importances_

def plot_importances(importances, col_array):
    # Calculate the feature ranking
    indices = np.argsort(importances)[::-1]    
    #Mean Feature Importance
    print "\nMean Feature Importance %.6f" %np.mean(importances)    
    #Plot the feature importances of the forest
    #plt.figure(figsize=(20,8))
    
    #plt.title("Feature importances")
    #plt.bar(range(len(importances)), importances[indices],color="gr", align="center")
    #plt.xticks(range(len(importances)), col_array[indices], fontsize=14, rotation=90)
    #plt.xlim([-1, len(importances)])
    
    #plt.show()
    sns.barplot(x=col_array[indices],y=importances[indices])
    plt.xticks(rotation=90,fontsize=14)
    plt.yticks(fontsize=13)

def select_feature(data,feature_cols,importances):
    indices = np.argsort(importances)[::-1]    
    f_count=22
    f_start=np.int(np.sqrt(f_count))
    f_range=range(f_start,f_count)
    score=np.array(np.zeros(f_count-f_start))
    for i in f_range:
        cols=feature_cols[indices]
        cols=cols[:i]
        X_train,y_train=split_data(data,cols)
        cv=cv_score(create_clf(),X_train,y_train)
        print i       
        print cols
        print cv
        score[i-f_start]=cv
    return pd.DataFrame(score,index=f_range,columns=['cv_score'])
    

train=load_data('train.csv')
train= preprocess_data(train)
train=feature_engineering(train)
feature_cols= [col for col in train.columns if col not in train.columns[11:55] and col not in ['Cover_Type','Id']]
X_train,y_train=split_data(train,feature_cols)
importances=feature_importances(X_train,y_train)
plot_importances(importances,X_train.columns)
scores=select_feature(train,X_train.columns,importances)
print scores
scores.plot()