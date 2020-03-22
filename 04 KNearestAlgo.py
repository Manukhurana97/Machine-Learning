# classification algo
 #    1) k nearest neighbour algo -> pattern recognision 

import numpy as np
import pandas as pd
from collections import Counter
import warnings
import random


# 18,19

def KNearest_neighbors(data, predict, k=3):
    if (len(data)>=k):
        warnings.warn('k has less value')

    distance = []
    for group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict))
            distance.append([euclidean_distance,group])

    votes=[i[1] for i in sorted(distance)[:k]]
    #print(Counter(votes).most_common(1))
    votes_result=Counter(votes).most_common(1)[0][0]

    confidence = Counter(votes).most_common(1)[0][1] / k
    #print(votes_result, confidence)
    return votes_result, confidence

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)  # outliner
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()  # change string data to float
random.shuffle(full_data)  # shuffle data at random

# over version if train test,split

test_size=0.2
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-int(test_size*len(full_data))] # (.2 * length of full data) 80% of train data
test_data = full_data[-int(test_size*len(full_data)):] # last 20% of data

for i in train_data:  # populating the dictionaies
    train_set[i[-1]].append([i[:-1]]) # -ve 1 first element(last value) in the list
#   train_set[i[-1]] is to find type ,append data into list

for i in test_data:
    test_set[i[-1]].append([i[:-1]]) # -1 for last element of list, appending list into test_set  list till last element
#   test_set[i[-1]] is to find type ,append data into list

correct=0
total=0

for group in test_set:
    for data in test_set[group]:
        votes,confidence = KNearest_neighbors(train_set, data, k=5)
        if group == votes:
            correct += 1
        # else:
        #     print(confidence) # confidence votes that are incorrect
        total += 1
print(' Accuracy', correct/total)


