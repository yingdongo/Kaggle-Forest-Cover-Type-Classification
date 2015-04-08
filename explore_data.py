# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 17:23:41 2015

@author: Ying
"""
from tools import load_data
import ggplot as gp

def show_data(data):
    print data.head(10)
    #have a look at few top rows
    print data.describe()
    #describe() function would provide count, mean, standard deviation (std), min, quartiles and max in its output
def plotHist(data, x,c):
    picsPath = ""
    p = gp.ggplot(gp.aes(x=x,fill=c), data=data)
    p = p + gp.geom_histogram()
    p = p + gp.ggtitle("Histogram - %s" % (str(x)))
    gp.ggsave(p, "%stemp/Histogram-%s.png" % (picsPath, str(x)))


def plotPoints(data, x, y,c):
    picsPath = ""
    p = gp.ggplot(gp.aes(x=x, y=y,colour=c), data=data)
    p = p + gp.geom_point()
    p = p + gp.ggtitle("Scatter - %s vs. %s" % (str(x), str(y)))
    gp.ggsave(p, "%stemp/Scatter-%s_vs_%s.png" % (picsPath, str(x), str(y)))
def main():   
    train=load_data('train.csv')
    cols = [col for col in train.columns if col not in train.columns[11:56] and  col not in ['Id']] 
    xCol = 'Cover_Type'
    for col in cols:
        plotHist(train, col,xCol)
    yCols = [col for col in train.columns if col not in train.columns[11:56] and col not in ['Id', 'Cover_Type']] 
    for i in range(len(yCols)):
         for j in range(len(yCols)):
             if i!=len(yCols)-1-j:
                 plotPoints(train, yCols[i], yCols[len(yCols)-1-j],xCol)


if __name__ == '__main__':
    main()

