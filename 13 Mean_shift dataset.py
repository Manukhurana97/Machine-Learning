 # https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls

import numpy as np
import pandas as pd
from matplotlib import style
from sklearn.cluster import MeanShift
import matplotlib.pyplot as plt
from sklearn import preprocessing

style.use('ggplot')
df = pd.read_excel('titanic.xls')

original_df = pd.DataFrame.copy(df)


df.drop(['name', 'body'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
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
#print(df.head())

df.drop(['ticket', 'boat'], 1, inplace=True)

X = np.array(df.drop(['survived'], 1).astype(float)) # convert to float
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_
print(cluster_centers,labels)

# new column to our original dataframe:
original_df['cluster_group'] = np.nan

# iterate through the labels and populate the labels to the empty column
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i] # .iloc[i] row in df

n_clusters_ = len(np.unique(labels))
survival_rates = {}

for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] == float(i))]

    survival_cluster = temp_df[(temp_df['survived'] == 1)]

    survival_rate = len(survival_cluster) / len(temp_df)

    survival_rates[i] = survival_rate

print(survival_rates)



# # compare the group
#
# correct = 0
# for i in range(len(X)):
#     predict_me = np.array(X[i].astype(float))
#     predict_me = predict_me.reshape(-1, len(predict_me))
#     prediction = clf.predict(predict_me)
#
#     if prediction[0] == y[i]:
#         correct += 1
#
# print(correct/len(X))
