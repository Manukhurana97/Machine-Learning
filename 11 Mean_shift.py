# Its a hierarchical clustering algo
# machine figure out how many cluster are there and where they are.


import numpy as np
from sklearn.cluster import MeanShift
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import style
style.use("ggplot")

center = [[1, 1, 1], [5, 5, 5], [3, 10, 10]]
X, _ = make_blobs(n_samples=100, centers=center, cluster_std=1.5)


ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_center = ms.cluster_centers_

print(cluster_center)
n_cluster = len(np.unique(labels))
print('Number of estimate cluster', n_cluster)

colors = 10*['r', 'g', 'b', 'c', 'k', 'y', 'm']
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(X)):
    ax.scatter(X[i][0], X[i][1], X[i][2], c=colors[labels[i]], marker='o')

ax.scatter(cluster_center[:, 0], cluster_center[:, 1], cluster_center[:, 2], marker='x', color='k', s=150,zorder=10)

plt.show()
