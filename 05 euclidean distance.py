from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
style.use('fivethirtyeight')

# p=[2,3]
# q=[5,7]
# euclidean_distance = sqrt((q[0]-p[0])**2+(q[1]-p[1])**2)
# print(euclidean_distance)


dataset={'k':[[1,2],[2,3],[3,1]] ,'r':[[6,5,],[7,7,],[8,6]]}
new_features=[5,7]
#
# for i in dataset:
#     for data in dataset[i]:
#         plt.scatter(data[0],data[1] , s=100, color=i)
#
# plt.scatter(new_features[0],new_features[1])
# plt.show()


def K_Nearest_neighbors(data,predict,k=3):
    if len(data)>=k:
        warnings.warn('k has les values!')

    distances=[]
    for group in data:
        for feature in data[group]:
            euclidean_distance = sqrt((feature[0]-predict[0])**2+(feature[1]-predict[1])**2)
            #print(euclidean_distance)
            distances.append([euclidean_distance,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    votes_result=Counter(votes).most_common(1)[0][0]
    return votes_result

result = K_Nearest_neighbors(dataset,new_features, k=23)
print(result)

for i in dataset:
    for data in dataset[i]:
        plt.scatter(data[0],data[1] , s=100, color=i)

plt.scatter(new_features[0],new_features[1] ,color=result)
plt.show()
