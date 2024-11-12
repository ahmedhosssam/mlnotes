import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd

df = pd.read_csv('data/clustered_data.csv')

X = df['x'].to_numpy()
Y = df['y'].to_numpy()

C = np.full(X.size, 1)

clust_n = 5;

cluster_x = np.random.randint(np.min(X), np.max(Y), size=clust_n)
cluster_y = np.random.randint(np.min(X), np.max(Y), size=clust_n)

'''
K-Means Clustering Algorithm:

We want to initialize `C` of size X to assign each data point to the closest cluster.

Repeat until convergence:
    for i in range(X.size):
        # get the closest cluster centroid.
        C[i] = get_closest_centroid(X[i], Y[i]) <-- returns j (the index of the centroid)

    for j in range(clusters.size):
        cluster_x[j] = the x mean of the assigned cluster j in `C`.
        cluster_y[j] = the y mean of the assigned cluster j in `C`.
'''

def get_closest_centroid(x, y):
    min_d = 1e9
    c = 0
    for j in range(clust_n):
        d = (math.pow(cluster_x[j]-x, 2)+math.pow(cluster_y[j]-y, 2))
        if d < min_d:
            min_d = d
            c = j
    return c

def get_mean(j, axis):
    den = 0
    nom = 0
    if axis == 'x':
        den = np.count_nonzero(C==j)
        nom = np.sum(X[C==j])
    else:
        den = np.count_nonzero(C==j)
        nom = np.sum(Y[C==j])
    return nom/den if den != 0 else 0

for g in range(10000): # It should be: Repeat until convergence
    for i in range(C.size):
        # get the closest cluster centroid.
        C[i] = get_closest_centroid(X[i], Y[i])

    for j in range(clust_n):
        cluster_x[j] = get_mean(j, 'x')
        cluster_y[j] = get_mean(j, 'y')

plt.figure(figsize=(8, 6))
plt.scatter(X, Y, c='blue', alpha=0.6, edgecolors='k')

# plot the cluster centroids
print(cluster_x,cluster_y)
plt.scatter(cluster_x, cluster_y, color='red', marker='x')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clustered Data Points')
plt.show()
