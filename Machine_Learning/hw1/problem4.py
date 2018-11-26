import numpy as np
from matplotlib import pyplot as pl

new_data = np.loadtxt("DanWood.dt")

#Problem 1
Xs = new_data[:, 0]
Ys = new_data[:, 1]
result = np.vstack((np.ones(Xs.shape[0]), Xs))
X = result.T
Y = Ys.T

#Assumes the data is numpy arrays
def linearRegression(X, y):
  inv =  np.linalg.inv(np.matmul(np.transpose(X), X))
  return np.matmul(np.matmul(inv, np.transpose(X)), y)

#Problem 2
model = linearRegression(X, Y)

#lambda of the regression equation from our model
f = lambda x : model[0] + model[1] * x

predictions = [f(x) for x in Xs]


def error(l1, l2):
  err = 0.0
  for i in range(len(l1)):
    err += (l1[i] - l2[i]) * (l1[i] - l2[i])
  return err / len(l1)



print("The results from the linear regression are listed below")
print("The paramters are " + np.array2string(model, precision = 5))
print("The Mean Squared Error is " + str(float(error(predictions, Y))))

#Problem 3
"""
pl.scatter(Xs, Y)
pl.plot(Xs, f(Xs))
pl.show()
"""


#Problem 4
def mean(y):
  total = 0.0
  for v in y:
    total += v
  return total / len(y)

def variance(y):
  total = 0.0
  ave = mean(y)
  for v in y:
    total += (v - ave) * (v - ave)
  return total / len(y) #Should this be (len(y) - 1)

print("The Variance of the labels is " + str(variance(Y).item()))

#Problem 5

X3 = np.power(X, 3)
model3 = linearRegression(X3, Y)
f3 = lambda x : model3[0] + model3[1] * x * x * x
predictions3 = [f3(x) for x in Xs]

print("Results for the transformen mapping ")


print("The paramters are " + np.array2string(model3, precision = 5))
print("The Mean Squared Error is " + str(float(error(predictions3, Y))))



"""
fig1 = pl.figure(1)
pl.scatter(Xs, Y)
pl.plot(Xs, f(Xs))
fig2 = pl.figure(2)
pl.scatter(Xs, Y)
pl.plot(Xs, f3(Xs))
pl.show()
"""
