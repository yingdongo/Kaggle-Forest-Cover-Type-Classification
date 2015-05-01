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
    plotHist(train,'Elevation','Cover_Type')
#     xCol = 'Cover_Type'
#     for col in cols:
#         plotHist(train, col,xCol)
#     yCols = [col for col in train.columns if col not in train.columns[11:56] and col not in ['Id', 'Cover_Type']] 
#     for i in range(len(yCols)):
#          for j in range(len(yCols)):
#              if i!=len(yCols)-1-j:
#                  plotPoints(train, yCols[i], yCols[len(yCols)-1-j],xCol)
#==============================================================================


if __name__ == '__main__':
    main()

