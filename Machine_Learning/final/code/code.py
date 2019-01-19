
#Verbose for Debugging
DEBUG = True
def log(message):
  if DEBUG:
    print(



import numpy as np
from matplotlib import pyplot as pl


# Question 1 Starts here

trainInput = np.loadtxt(open("trainInput.csv", "rb"), delimiter=",")
trainTarget = np.loadtxt(open("trainTarget.csv", "rb"), delimiter=",")

print(trainInput.shape)
print(trainTarget.shape)


