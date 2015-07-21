# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:34:44 2015

@author: Ying
"""
from sklearn.metrics import confusion_matrix
from sklearn import ensemble

from feature_engineering import feature_engineering
from data_preprocess import fill_missing
from data_preprocess import preprocess_data
from feature_selection import get_features
from tools import split_data
from tools import load_data
from tools import cv_score
import numpy as np
from sklearn.cross_validation import ShuffleSplit

def get_data():
    train=load_data('train.csv')
    test=load_data('test.csv')
    train=preprocess_data(train)
    test=preprocess_data(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    
    X_train,y_train=split_data(train,feature_cols)
    feature_cols=get_features(X_train,y_train)
    return feature_cols,train,test

def get_clf():
    clf=ensemble.ExtraTreesClassifier(max_features= 0.35, n_estimators= 650)
    return clf
    
def get_clf1():
    clf=ensemble.RandomForestClassifier(max_features= 'log2', n_estimators= 400)
    return clf
        
def get_ef():
    clf=ensemble.ExtraTreesClassifier(n_estimators= 100)
    return clf    
    
def get_rf():
    forest=ensemble.RandomForestClassifier(n_estimators=100)
    return forest
    
def get_confusion_t(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm

def basicclf_predict(X_train,y,X_test):
    y_pred=get_ef().fit(X_train,y).predict(X_test)
    return y_pred
    
def clf_predict(clf,X_train,y,X_test):
    y_pred=clf.fit(X_train,y).predict(X_test)
    return y_pred

def write_result(test_ids,y_pred,filename):
    with open(filename, "wb") as outfile:
       outfile.write("Id,Cover_Type\n")
       for e, val in enumerate(list(y_pred)):
           outfile.write("%s,%s\n"%(test_ids[e],val))
           
def sep_clf(clf,train,X_test,feature_cols):   
    seq=np.array([7,6,5,4])
    #pres=np.ndarray(shape=(7,len(test)))
    pres=[]
    lens=[]
    ids=np.array(np.arange(0, len(X_test)))

    for i in seq:
        train_num=train.copy()
        train_num['Cover_Type'][train_num['Cover_Type']<>i]=0
        X_train = train_num[feature_cols]
        y = train_num['Cover_Type']
        clf.fit(X_train,y)
        pre=clf.predict(X_test)
        id_a=np.where(pre==i)
        id_i=ids[pre==i]
        pres=np.append(pres,id_i)
        lens=np.append(lens,len(id_i))
        X_test=X_test.drop(X_test.index[id_a])
        ids=ids[pre<>i]
    
    train_num=train.copy()
    train_num=train_num[train_num['Cover_Type']<4]
    X_train = train_num[feature_cols]  
    y = train_num['Cover_Type']
    clf.fit(X_train,y)
    pre=clf.predict(X_test)
    
    id_i=ids[pre==3]
    pres=np.append(pres,id_i)
    lens=np.append(lens,len(id_i))
    
    id_i=ids[pre==2]
    pres=np.append(pres,id_i)
    lens=np.append(lens,len(id_i))
    
    id_i=ids[pre==1]
    pres=np.append(pres,id_i)
    lens=np.append(lens,len(id_i))
    
    lens=lens.astype(int)
    pres=pres.astype(int)
    ctype=np.array(np.zeros(len(pres)),dtype=int)
    running_sum=np.cumsum(lens)
    
    ctype[0:running_sum[0]]=7
    ctype[running_sum[0]-2:running_sum[1]]=6
    ctype[running_sum[1]-2:running_sum[2]]=5
    ctype[running_sum[2]-2:running_sum[3]]=4
    ctype[running_sum[3]-2:running_sum[4]]=3
    ctype[running_sum[4]-2:running_sum[5]]=2
    ctype[running_sum[5]-2:running_sum[6]]=1
    p=pres.argsort()
    y_pred=ctype[p]
    return y_pred
    
def write_submit(test_ids,y_pred): 
    with open('newsubmission/algorithm_solution7654_0.35_650.csv', "wb") as outfile:
        outfile.write("Id,Cover_Type\n")
        for k in range(0,len(test_ids)):
            outfile.write("%s,%s\n"%(test_ids[k],y_pred[k]))

def simple_solution(train,test):
    cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train=train[cols]
    y=train['Cover_Type']
    X_test=test[cols]
    test_ids=test['Id']
    score=cv_score(get_rf(),X_train,y)
    print 'simple_solution with random forest:'
    print score
    y_pred=clf_predict(get_rf(),X_train,y,X_test)
    write_result(test_ids,y_pred,'newsubmission/simple_solution_rf100_total.csv')
#cv 0.784788359788 leaderbord 0.75099 771
def filling_solution(train,test):
    train=fill_missing(train)
    test=fill_missing(test)
    cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train=train[cols]
    y=train['Cover_Type']
    X_test=test[cols]
    test_ids=test['Id']
    print 'fill_in_missing_solution with random forest:'
    score=cv_score(get_rf(),X_train,y)
    print score
    y_pred=clf_predict(get_rf(),X_train,y,X_test)
    write_result(test_ids,y_pred,'newsubmission/fill_in_missing_solution.csv')
#0.780423280423 Leaderbord 0.75301 715
    
def preprocess_solution(train,test):
    train=preprocess_data(train)
    test=preprocess_data(test)
    cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train=train[cols]
    y=train['Cover_Type']
    X_test=test[cols]
    test_ids=test['Id']
    score=cv_score(get_rf(),X_train,y)
    print 'preprocess_solution with random forest:'
    print score
    y_pred=clf_predict(get_rf(),X_train,y,X_test)
    write_result(test_ids,y_pred,'newsubmission/preprocess_solution.csv')    
#0.776917989418 Leaderbord 0.75084 774
    
def engineering_solution(train,test):
    train=preprocess_data(train)
    test=preprocess_data(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train=train[cols]
    y=train['Cover_Type']
    X_test=test[cols]
    test_ids=test['Id']
    score=cv_score(get_rf(),X_train,y)
    print 'engineering_solution with random forest:'
    print score
    y_pred=clf_predict(get_rf(),X_train,y,X_test)
    write_result(test_ids,y_pred,'newsubmission/engineering_solution.csv')  
#0.807142857143 Leaderbord 0.78359 316

def selection_solution(train,test):
    train=preprocess_data(train)
    test=preprocess_data(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train=train[cols]
    y=train['Cover_Type']
    feature_cols=get_features(X_train,y)
    X_train=train[feature_cols]
    X_test=test[feature_cols]
    test_ids=test['Id']
    score=cv_score(get_rf(),X_train,y)
    print 'selection_solution:'
    print score
    y_pred=clf_predict(get_rf(),X_train,y,X_test)
    write_result(test_ids,y_pred,'newsubmission/selection_solution26.csv')    
#60 features 0.807076719577 Leaderboard 0.78370
#feature selection 0.810119
#43 features 0.805291005291 Leaderboard 0.78113
#26 features 0.797023809524 Leaderboard  0.76790

def model_solution(train,test):
    train=preprocess_data(train)
    test=preprocess_data(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id','Vertical_Distance_To_Hydrology','Slope','Soil_Type7','Soil_Type8','Soil_Type15']]
    X_train=train[feature_cols]
    y=train['Cover_Type']
    #feature_cols=get_features(X_train,y)
    X_train=train[feature_cols]
    X_test=test[feature_cols]
    test_ids=test['Id']
    score=cv_score(get_ef(),X_train,y)
    print 'model_solution_slopeandvdth:'
    print score
    y_pred=clf_predict(get_ef(),X_train,y,X_test)
    write_result(test_ids,y_pred,'newsubmission/model_solution_slopeandvdth.csv')   
#0.81746031746 Leaderboard 0.80767
#without slope 0.818518518519   0.80841
#without slope and Ele_plus_VDtHyd 0.817526455026
#untder total 0.817592592593
# with aspect 0.819312169312 Leaderboard 0.80888
#without slope and vertical distance tohydrology 0.818518518519 Leaderboard 0.80986
    
def paramtuing_solution(train,test):
    train=preprocess_data(train)
    test=preprocess_data(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    #feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id','Vertical_Distance_To_Hydrology','Slope','Soil_Type7','Soil_Type8','Soil_Type15']]
    X_train=train[feature_cols]
    y=train['Cover_Type']
    X_test=test[feature_cols]
    test_ids=test['Id']
    score=cv_score(get_clf1(),X_train,y)
    print 'paramtuing_solution:'
    print score
    y_pred=clf_predict(get_clf1(),X_train,y,X_test)
    write_result(test_ids,y_pred,'newsubmission/paramtuing_solution_rf_400log2.csv')    
    #extratrees 600 0.35 0.822023809524 leaderboard 0.81256
    #random forest 0.812301587302 leaderboard 0.78871

def algorithm_solution(train,test):
    train=preprocess_data(train)
    test=preprocess_data(test)
    train=feature_engineering(train)
    test=feature_engineering(test)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id','Vertical_Distance_To_Hydrology','Slope','Soil_Type7','Soil_Type8','Soil_Type15']]
    test_ids=test['Id']
    X_test = test[feature_cols]
    y_pred=sep_clf(get_clf(),train,X_test,feature_cols)
    write_submit(test_ids,y_pred)
    #leaderboard 0.82209
def validation():
    train=load_data('train.csv')
    train=preprocess_data(train)
    train=feature_engineering(train)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id','Vertical_Distance_To_Hydrology','Slope','Soil_Type7','Soil_Type8','Soil_Type15']]
    rs = ShuffleSplit(15120, n_iter=10, random_state=0)   
    scores=np.array(np.zeros(10))
    i=0
    for train_index, test_index in rs:   
        X_train=train.irow(train_index)
        test=train.irow(test_index)
        pred=sep_clf(get_clf(),X_train,test[feature_cols],feature_cols)
        compare=pred==test['Cover_Type']
        scores[i]=np.float(compare.sum())/np.float(len(compare))
        i=i+1
    return scores
    #0.89563492063492056 47563 0.82172
    #0.89556878306878307 76543 0.82209
    #0.90238095238095239 7654 0.82261 0.82269 with 650 0.35
    #0.90251322751322749 4756 0.82209

def main():
    train=load_data('train.csv')
    test=load_data('test.csv')
    #filling_solution(train.copy(),test.copy())
    #simple_solution(train.copy(),test.copy())
    #preprocess_solution(train.copy(),test.copy())
    #engineering_solution(train.copy(),test.copy())
    #selection_solution(train.copy(),test.copy())
    #model_solution(train.copy(),test.copy())
    #paramtuing_solution(train.copy(),test.copy())
    algorithm_solution(train.copy(),test.copy())
    
if __name__ == '__main__':
    main()    

#scores=validation()