# Clustering(UnSupervised Learning) :
# 1) flast
# 2) hierarichal
# semi supervised supervised learning

import numpy as np
from sklearn.cluster import KMeans
import matplotlib .pyplot as plt
from matplotlib import style
style.use('ggplot')


# dataset
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
plt.scatter(X[:, 0], X[:, 1], s=100, linewidth=1) # 0th and 1st element if x array, size=150, linewidths=5

# classifier
clf = KMeans(n_clusters=5) # 2 clusters
clf.fit(X)

centroid = clf.cluster_centers_
labels = clf.labels_  # y

colors=['g.', 'r.', 'c.',  'b.', 'k.'] # . is period (marker)

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10) #
plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', s=100, linewidths=2)
plt.show()
