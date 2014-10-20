# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 19:14:16 2014

@author: dhewitt
"""

#%%

from sklearn import datasets
iris = datasets.load_iris()

iris.data.shape
#%%

iris.target.shape
#%%

import numpy as np
np.unique(iris.target)

#%%
#PRINCIPAL COMPONENT ANALYSIS - 3 DIMENSIONS

from sklearn import decomposition

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

pca = decomposition.PCA(n_components=3)
pca.fit(iris.data)

X = pca.transform(iris.data)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:,1], X[:,2], c=iris.target)

#%%
#PRINCIPAL COMPONENT ANALYSIS - 2 DIMENSIONS


pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)

X = pca.transform(iris.data)

import pylab as pl
pl.scatter(X[:, 0], X[:, 1], c=iris.target) 

#%%
#SVMs FOR CLASSIFICATION

from sklearn import svm
svc = svm.SVC(kernel='linear')
svc.fit(X, iris.target)

#%%
minmax = np.array([X.min(axis=0), X.max(axis=0)]).transpose()

randoms = np.array([np.random.uniform(Xmin, Xmax, 100) for Xmin, Xmax in minmax]).transpose()
predicts = svc.predict(randoms)
pl.scatter(randoms[:,0], randoms[:,1], c=predicts)
    