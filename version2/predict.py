# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 10:34:44 2015

@author: Ying
"""
from sklearn.metrics import confusion_matrix
from sklearn import ensemble
from feature_engineering import add_feature
from explore_data import preprocess_data
from feature_selection import get_features
from feature_selection import split_data
import numpy as np
def get_data():
    train=preprocess_data('train.csv')
    test=preprocess_data('test.csv')
    add_feature(train)
    add_feature(test)
    feature_cols= [col for col in train.columns if col  not in ['Cover_Type','Id']]
    X_train,y_train=split_data(train,feature_cols)
    feature_cols=get_features(X_train,y_train)
    return feature_cols,train,test

def get_clf():
    clf=ensemble.ExtraTreesClassifier(max_features= 0.3, n_estimators= 500)
    return clf
    
def get_confusion_t(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm

def clf_predict(X_train,y,X_test):
    y_pred=get_clf().fit(X_train,y).predict(X_test)
    return y_pred

def write_result(test_ids,y_pred):
    with open('submit_extraforests1.csv', "wb") as outfile:
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
    with open('submit_extraforestsep7654.csv', "wb") as outfile:
        outfile.write("Id,Cover_Type\n")
        for k in range(0,len(test_ids)):
            outfile.write("%s,%s\n"%(test_ids[k],y_pred[k]))
            
def main():
    cols,train,test=get_data()
    #X_train=train[cols]
    #y=train['Cover_Type']
    X_test=test[cols]
    test_ids=test['Id']
    y_pred=sep_clf(get_clf(),train,X_test)
    write_submit(test_ids,y_pred)
    #y_pred=clf_predict(X_train,y,X_test)
    #write_result(test_ids,y_pred)

if __name__ == '__main__':
    main()    