import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
style.use('ggplot')

#dataset

X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [9, 8], [7, 7], [1, 10], [1, 0.6], [9, 11], [9, 3], [7, 11], [8,4],[4, 6]])

# plt.scatter(X[:, 0], X[:, 1], s=100, linewidth=1) # 0th and 1st element , size=150, linewidths=5
# plt.show()

colors = 10 * ['g.', 'r.', 'c.',  'b.', 'k.'] # . is period (marker)

class Mean_Shift:
    def __init__(self, radius=3):
        self.radius = radius

    def fit(self, data):
        centroids = {}

        # intial centroid
        for i in range(len(data)):
            centroids[i] = data[i] # key = centroid[i], value = data[i]

        while True:
            new_centroids = []

            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]  # i is key to get centroid

                 # iterate through data
                for featureset in data:
                    if np.linalg.norm(featureset - centroid) < self.radius:
                        in_bandwidth.append(featureset)

                new_centroid = np.average(in_bandwidth,axis=0) # mean vector  of average
                new_centroids.append(tuple(new_centroid))# array to tuple

            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)

            centroids = {}  # sorted centroid

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break

        self.centroids = centroids


clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids
plt.scatter(X[:, 0], X[:, 1], s=100, linewidth=1) # 0th and 1st element , size=150, linewidths=5

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()


