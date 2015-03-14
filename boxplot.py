# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:08:31 2015

@author: Ying
"""
from matplotlib import pyplot as plt
import numpy as np

def box_plot(data):  
    # basic plot
    plt.boxplot(data)  
    plt.show()