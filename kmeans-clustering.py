# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 10:39:38 2020

@author: chint
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv('Mall_Customers.csv')
x=data.iloc[:,[3,4]].values

#Elbow method for finding the appropriate number of clusters for our problem
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',n_init=10,max_iter=300,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.Figure()
plt.plot(range(1,11),wcss)
plt.title("wcss v/s number of clusters")
plt.xlabel('number of clusters')
plt.ylabel('wcss')
plt.show()

#best number of clusters is 5

kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(x)

plt.figure
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,color='red', label='cluster-1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,color='yellow', label='cluster-2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,color='orange', label='cluster-3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,color='green', label='cluster-4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,color='cyan', label='cluster-5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,color='black', label='centroid')
plt.title('visualization of clustering results')
plt.xlabel('annual income')
plt.ylabel('spemding score')
plt.show()

