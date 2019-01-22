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

