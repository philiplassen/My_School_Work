import numpy as np
from matplotlib import pyplot as pl

new_data = np.loadtxt("DanWood.dt")
print(new_data)

print("did we fail")

Xs = new_data[:, 0]
Ys = new_data[:, 1]
print(Xs)
print(Xs.shape[0])

result = np.vstack((np.ones(Xs.shape[0]), Xs))
print(result)
print(result.shape)
print("on to the next")
X = result.T
Y = Ys.T
print(X)
print(X.shape)



#Assumes the data is numpy arrays
def linearRegression(X, y):
  inv =  np.linalg.inv(np.matmul(np.transpose(X), X))
  return np.matmul(np.matmul(inv, np.transpose(X)), y)

print("our results from regression")
print(linearRegression(X, Y))

model = linearRegression(X, Y)

f = lambda x : model[0] + model[1] * x

predictions = [f(x) for x in Xs]


def error(l1, l2):
  err = 0
  for i in range(len(l1)):
    err += (l1[i] - l2[i]) * (l1[i] - l2[i])
  return err


print("Error")
print(error(predictions, Y))


print("The paramters are (" + str(float(model[0])) + ", " + str(float(model[1]))) + ")"
print("The Mean Squared Error is " + str(float(error(predictions, Y))))

pl.scatter(Xs, Y)
pl.plot(Xs, f(Xs))
pl.show()
