import matplotlib.pyplot as pl
import skimage.io as ski
import pylab as pyl
import numpy as np
import numpy.random as npr

def blend(A, B, w_A, w_B):
  return A * w_A + B * w_B

  
