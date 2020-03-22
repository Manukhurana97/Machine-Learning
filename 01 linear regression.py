# linear reqresion take the continous data and produce best fit line

import pandas as pd
import quandl
import math
import numpy as np # use arry
from sklearn import preprocessing,svm # svm ->support vector machine
from sklearn.model_selection import train_test_split # used for traing and testing samples
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
#print(df.head())
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
#print(df.head())



forcast_data = 'Adj. Close'
df.fillna(-99999, inplace=True) # fill-> fill,na->not avaliable (can work with na)

forcast_out = int(math.ceil(0.01*len(df))) # predict 10% of dataframe, math.ceil round  to the nearest whole

df['label']=df[forcast_data].shift(-forcast_out)
df.dropna(inplace=True)
#print(df.head())
#print(df.tail)

X=np.array(df.drop(['label'],1)) # features
y= np.array(df['label']) # label
X=preprocessing.scale(X) # we are scaling x before classifier

#X=X[:-forcast_out+1]    no need
df.dropna(inplace=True)
y=np.array(df['label'])
#print(len(X),len(y))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 0.2 means 20 %
clf = LinearRegression()
clf.fit(X_train, y_train)# fit for train
accuracy=clf.score(X_test, y_test) # score for text

#print(accuracy)


#clf=LinearRegression(n_jobs=1)
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train,y_train)
    accuracy=clf.score(X_test, y_test)
    print(accuracy)

