# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:43:44 2020

@author: chint
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Mall_Customers.csv')
x=data.iloc[:,[3,4]].values

#finding the appropriate number of clusters for our problem by using dendrograms
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title("euclidian distnce v/s points")
plt.xlabel('data points')
plt.ylabel('euclidian distance')
plt.show()

#best number of clusters is 5

from sklearn.cluster import AgglomerativeClustering
agc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=agc.fit_predict(x)

plt.figure
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,color='red',label='cluster-1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,color='yellow',label='cluster-2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,color='orange',label='cluster-3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,color='green', label='cluster-4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,color='cyan', label='cluster-5')
plt.title('visualization of clustering results')
plt.xlabel('annual income')
plt.ylabel('spemding score')
plt.legend()
plt.show()

