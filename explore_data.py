# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:23:41 2015

@author: Ying
"""
from tools import load_data
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
import seaborn as sns



def show_data(data):
    print data.head(10)
    #have a look at few top rows
    print data.describe()
    #describe() function would provide count, mean, standard deviation (std), min, quartiles and max in its output

def plotHist(data, x,c):
    pd.DataFrame.plot(data,x=c,y=x,kind='hist',legend=True)


def main():   
    train=load_data('train.csv')
    cols = [col for col in train.columns if col not in train.columns[11:56] and  col not in ['Id']] 
        #train[['Hillshade_3pm','Hillshade_9am']].boxplot()
    #plotHist(train,'Elevation','Cover_Type')
    #sns.lmplot(x="Hillshade_Noon",y="Hillshade_3pm",data=train,fit_reg=False,hue="Cover_Type", palette="husl",hue_order=[1,2,3,4,5,6,7])
    #sns.lmplot(x="Hillshade_9am",y="Hillshade_3pm",data=train,fit_reg=False,hue="Cover_Type", palette="husl",hue_order=[1,2,3,4,5,6,7])
    class1=train['Elevation'][train['Cover_Type']==1]
    class2=train['Elevation'][train['Cover_Type']==2]
    class3=train['Elevation'][train['Cover_Type']==3]
    class4=train['Elevation'][train['Cover_Type']==4]
    class5=train['Elevation'][train['Cover_Type']==5]
    class6=train['Elevation'][train['Cover_Type']==6]
    class7=train['Elevation'][train['Cover_Type']==7]
    d=[class1,class2,class3,class4,class5,class6,class7]
    d=np.transpose(d)
    da=pd.DataFrame(data=d,columns=[1,2,3,4,5,6,7])

    fig, ax =plt.subplots(figsize=(10,8))
    ax.hist(da.values,20, histtype='step', stacked=True, fill=True,label=da.columns)
    legend=ax.legend(fontsize=14,title='Cover_Type')
    plt.setp(legend.get_title(),fontsize=13)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel('Elevation',fontsize=16)
    #g = sns.FacetGrid(data=train, hue='Cover_Type', hue_order=[1,2,3,4,5,6,7],size = 8)
    #g.map(sns.distplot, 'Elevation', hist_kws={'histtype':'step','stacked':True,'fill':True})
    #g.add_legend(fontsize=14)
    #plt.xticks(fontsize=13)
    #plt.xlabel('Elevation',fontsize=14)
    

if __name__ == '__main__':
    main()

