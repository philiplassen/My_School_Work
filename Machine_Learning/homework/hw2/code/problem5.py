import numpy as np
import math
import random


def logistic_regression(ys, xs):
  w = np.zeros(len(xs[0]))
  N = len(xs)
  for t in range(10000):
    if (t % 10 == 0):
      print("Currently on the " + str(t) + "th iteration")
    total = np.zeros(len(xs[0]))
    for n in range(N):
      total += ys[n] * xs[n] / (1 + math.exp(np.matmul(w, xs[n]) * ys[n]))
    w = w + (1.0 / N) * total * 0.001
    print(w)
  return w
  
def sig(v):
  return math.exp(x) / (1 + math.exp(x))



