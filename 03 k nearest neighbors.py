 # classification algo
 #    1) k nearest neighbour algo -> pattern recognision for cancer

import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

df=pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True) # replace ? , -99999 is consider as outliner for dumping the data
df.drop(['id'], 1, inplace=True)# remove id col

X = np.array(df.drop(['class'], 1)) # X= feature :- Every things except class column
y = np.array(df['class'])# Y= label :- class column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier () # call the classifier
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)


# example
example_measure=np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [8, 10, 10, 8, 7, 10, 9, 7, 1], [1, 1, 1, 1, 1, 1, 3, 1, 1]])

example_measure=example_measure.reshape(len(example_measure),-1)
prediction=clf.predict(example_measure)
print(prediction)
