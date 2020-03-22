# 42
import random
import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_blobs
style.use('ggplot')

#dataset

# X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [9, 8], [7, 7], [1, 10], [1, 0.6], [9, 11], [9, 3], [7, 11], [8,4],[4, 6]])

# plt.scatter(X[:, 0], X[:, 1], s=100, linewidth=1) # 0th and 1st element , size=150, linewidths=5
# plt.show()

centers = random.randrange(4, 8)

X,y = make_blobs(n_samples=10 , centers=centers, n_features=2)

colors = 10 * ['g.', 'r.', 'c.',  'b.', 'k.'] # . is period (marker)

class Mean_Shift:
    def __init__(self, radius=None, radius_norm_step=50):

        self.radius_norm_step = radius_norm_step
        self.radius = radius

    def fit(self, data):

        if self.radius == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid) # Magnitude from origan
            self.radius = all_data_norm/self.radius_norm_step




        centroids = {}

        # intial centroid
        for i in range(len(data)):
            centroids[i] = data[i] # key = centroid[i], value = data[i]

        while True:
            new_centroids = []

            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]  # i is key to get centroid

                # weights
                weights= [i for i in range (self.radius_norm_step)][::-1] # [::1] reverse the list


                 # iterate through data
                for featureset in data:



                    distance = np.linalg.norm(featureset - centroid)
                    if distance == 0:
                        distance= 0.0001
                    weights_index = int(distance/self.radius) # total radius step, more step less weight

                    if weights_index>self.radius_norm_step-1: # if weight index is greater the max distance
                        weights_index = self.radius_norm_step-1 # weight index = max distance

                    to_add = (weights[weights_index]**2)*[featureset]
                    in_bandwidth += to_add


                new_centroid = np.average(in_bandwidth,axis=0) # mean vector  of average
                new_centroids.append(tuple(new_centroid))# array to tuple

            uniques = sorted(list(set(new_centroids)))

            to_pop=[]

            for i in uniques:
                for ii in uniques:
                    if i ==ii:
                        pass
                    elif np.linalg.norm(np.array(i)-np.array(ii))<=self.radius:
                        to_pop.append(ii)
                        break

            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass



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

        self.classifications = {}
        for i in range(len(self.centroids)):
            self.classifications[i] = []
        for featureset in data:
            distance= [np.linalg.norm(featureset-self.centroids[centroids]) for centroids in self.centroids]
            classification= distance.index(min(distance))
            self.classifications[classification].append(featureset)



    def predict(self,data):
        distance = [np.linalg.norm(featureset - self.centroids[centroids]) for centroids in self.centroids]
        classification = distance.index(min(distance))
        return classification


clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

#plt.scatter(X[:, 0], X[:, 1], s=100, linewidth=1) # 0th and 1st element , size=150, linewidths=5

for classification in clf.classifications:
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], marker="x", color='r', s=150, linewidths=5, zorder=10)



for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()


