# k Means clusterings(custom)

import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
from matplotlib import style
from sklearn.model_selection import train_test_split
style.use('ggplot')



# dataset
X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
plt.scatter(X[:, 0], X[:, 1], s=100, linewidth=1) # 0th and 1st element , size=150, linewidths=5
plt.show()


colors = ['g.', 'r.', 'c.',  'b.', 'k.'] # . is period (marker)

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter = 300): # tol is how much centroid move in  percentage change of, max_iter = no of time to run
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):
        self.centroid = {}

        for i in range(self.k):
            self.centroid[i] = data[i] # iterating through data X ,
            # first 2 centroid [1, 2], [1.5, 1.8]

        for i in range(self.max_iter):
            self.classifications = {} # centroid and claffification
            # keys are centroid and values are  fearures set

            for i in range(self.k):
                self.classifications[i] = [] #

            for feature_set in data:
                # calculate the distances
                distances=[np.linalg.norm(feature_set-self.centroid[centroid]) for centroid in self.centroid]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature_set)

            prev_centroid = dict(self.centroid)

            for classification in self.classifications:
                self.centroid[classification] = np.average(self.classifications[classification], axis=0)# average of centroid, redefines the centroid

            optimized = True

            for c in self.centroid:
                original_centroid = prev_centroid[c]
                curreny_centoid = self.centroid[c]

                if np.sum((curreny_centoid-original_centroid)/original_centroid*100.0)>self.tol:
                    print(np.sum((curreny_centoid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroid[centroid]) for centroid in self.centroid]
        classification = distances.index(min(distances))

        return classification

# Clustering non numeric data

df = pd.read_excel('titanic.xls')
df.drop(['name', 'body'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
#print(df.head())


def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_content = df[column].values.tolist() # convert to list
            unique_element = set(column_content) # non repetative values
            x = 0
            for unique in unique_element:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

            df[column] = list(map(convert_to_int, df[column])) # reseting the value of list by mapping to the convert_to_int

    return df


df = handle_non_numerical_data(df)
print(df.head())

# add/remove features just to see impact they have.
df.drop(['ticket', 'home.dest'], 1, inplace=True)
X = np.array(df.drop(['survived'], 1).astype(float))
y = np.array(df['survived'])

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2)


###############

clf = K_Means()
clf.fit(X)
#
# for centroid in clf.centroid:
#      plt.scatter(clf.centroid[centroid][0], clf.centroid[centroid][1], marker=0, s=150, color='k')
#
# for classification in clf.classifications:
#     colors = colors[classification]
#
#     for feature_set in clf.classifications[classification]:
#         plt.scatter(feature_set[0], feature_set[0], marker='x', s=150, linewidths=2)
#
#
# unknown = np.array([[1, 4],
#                    [3, 2],
#                    [2, 5],
#                    [8, 1],
#                    [7, 7]])
#
# for uk in unknown:
#     classification = clf.predict(uk)
#     plt.scatter(uk[0], uk[1], marker='*', s=150, linewidths=3)
# plt.show()

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[1].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    predict_me = clf.predict(predict_me)

    if predict_me == y[i]:
        correct+=1

print(correct/len(X))
