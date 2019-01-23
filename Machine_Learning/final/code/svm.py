import sys
DEBUG = False
for arg in sys.argv:
  if arg == '-v':
    DEBUG = True

#Verbose for Debugging
def log(message):
  if DEBUG:
    print(message)

import numpy as np
import pandas as pd

fileX = "trainInput.csv"
filey = "trainTarget.csv"


X = np.loadtxt(open("trainInput.csv", "rb"), delimiter=",")
y = np.loadtxt(open("trainTarget.csv", "rb"), delimiter=",")


index0s = [i for i in range(len(y)) if y[i] == 0]
index1s = [i for i in range(len(y)) if y[i] == 1]

log(len(index0s))
log(len(index1s))
log(len(y))

from sklearn import svm
from sklearn.model_selection import GridSearchCV


# Need a method for generating G given X and y
def G(X, y):
  index0s = [i for i in range(len(y)) if y[i] == 0]
  index1s = [i for i in range(len(y)) if y[i] == 1]
  X0 = X[index0s, :]
  X1 = X[index1s, :]
  

  G = []
  for i in range(len(y)):
    G += [shortestDistance(X[i, :], X1)] if y[i] == 0 else [shortestDistance(X[i, :], X0)]
  log(G[len(G) - 1])
  log(G[0])
  return np.median(G)


def shortestDistance(x, X):
  distances = [np.linalg.norm(x-xi) for xi in X]
  distances.sort()
  return distances[0]

theta = G(X, y)
gamma = 1.0 / (2 * (theta ** 2))

print(theta)
print(gamma)

b = 2


gammas = [gamma * (b ** i) for i in range(-3, 4)]
Cs = [b ** i for i in range(-1, 4)]
