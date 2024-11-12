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
    c, min_d = min(((j, (cluster_x[j] - x)**2 + (cluster_y[j] - y)**2) for j in range(clust_n)), key=lambda item: item[1])
    return c

def get_mean(j, axis):
    den, nom = np.count_nonzero(C==j), np.sum((X if axis == 'x' else Y)[C==j])
    return nom/den if den != 0 else 0

for g in range(10000): # It should be: Repeat until convergence
    # get the closest cluster centroid. 
    for i in range(C.size): C[i] = get_closest_centroid(X[i], Y[i])
    for j in range(clust_n):
        cluster_x[j], cluster_y[j] = get_mean(j, 'x'), get_mean(j, 'y')

plt.figure(figsize=(8, 6))
plt.scatter(X, Y, c='blue', alpha=0.6, edgecolors='k')

# plot the cluster centroids
print(cluster_x,cluster_y)
plt.scatter(cluster_x, cluster_y, color='red', marker='x')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clustered Data Points')
plt.show()
