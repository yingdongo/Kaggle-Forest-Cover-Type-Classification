# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 18:34:01 2015

@author: Ying
"""

import matplotlib.pyplot as plt

scores=[0.75614,0.75754,0.80845,0.81096,0.81092,0.81944]
names=["simple","preprocess","feature_engineering","parameter_tuning","feature_selection","algorithm"]
rankings=[557,527,105,73,74,24]

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_xticklabels(names,rotation='vertical')
plt.plot(scores)
plt.show()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
ax.set_xticklabels(names,rotation='vertical')
plt.plot(rankings)
plt.show()