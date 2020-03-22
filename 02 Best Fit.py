    # __Best fit slope__

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs=np.array([1,2,3,4,5,6] ,dtype=np.float64)
#ys=np.array([5,9,4,3,7,6],dtype=np.float64)

def create_dataset(hm, variance,step=2 , correlation=False): # hm-> how many, variance->how variable we want for data set, step-> how far from stop in y value, correlation -> for +,- (from step value)
    val = 1
    ys = []
    for i in range(hm):
        y = val+random.randrange(-variance,+variance)
        ys.append(y)
        if((correlation and correlation)=='pos'):
            val+=step
        elif (correlation and correlation)=='neg':
            val-=step
    xs=[i for i in range(len(ys))]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope_and_intercept(xs,ys):
    # __formula for best fit line (m)__
    m=(((mean(xs)*mean(ys))-mean(xs*ys)) /
       ((mean(xs)**2) - mean(xs*xs)))


    # __formula for best fit (y) intercept__
    b = (mean(ys) - m * mean(xs))
    return m,b

# its used to determine (how good is the best line)
def squared_error(ys_orig,ys_line):#its a distance between line and the points(and square)
    return sum((ys_line - ys_orig)*(ys_line - ys_orig))


def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


xs, ys = create_dataset(40, 80, 2, correlation='pos')


m, b = best_fit_slope_and_intercept(xs,ys)


#print(m,b)


regression_line = [(m*x)+b for x in xs]

#or
#regression_line=[]
#for x in xs:
#    regression_line.append((m*x)+b)

#__predeciton__

predict_x = 8
predict_y = (m*predict_x)+b

#print(predict_y)

r_square = coefficient_of_determination(ys, regression_line)
print(r_square)


plt.scatter(xs, ys)
plt.scatter(predict_y, predict_y, color='y', s=100)
plt.plot(xs, regression_line)
plt.show()
