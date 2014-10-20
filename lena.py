# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 16:30:03 2014

@author: dhewitt
"""

#%%
#IMAGE COMPRESSION USING CLUSTERING

import numpy as np

from scipy import misc
from sklearn import cluster
lena = misc.lena().astype(np.float32)
X = lena.reshape((-1, 1)) # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(n_clusters=8)
k_means.fit(X)
values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
lena_compressed = np.choose(labels, values)
lena_compressed.shape = lena.shape
#%%

import matplotlib.pyplot as plt
import matplotlib.cm as cm

plt.figure()
plt.imshow(lena, cmap=cm.Greys_r)

#%%
plt.figure()
plt.imshow(lena_compressed, cmap=cm.Greys_r)