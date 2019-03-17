import numpy as np
import matplotlib.pyplot as plt
def J(x, y, sigma):
  total = 0
  vals = np.arange(-5, x, 0.01)
  for v in vals:
    total += (1 / (2 * np.pi * np.square(sigma))) * np.exp( - (np.square(v) / (2 * np.square(sigma)))) * 0.01
  return total

def G(x, y,  sigma):
  return (1 / (2 * np.pi * np.square(sigma))) * np.exp( - ((np.square(x) + np.square(y))/ (2 * np.square(sigma))))

xs = np.linspace(-5, 5, 101)
z = [J(v, 1, 1) for v in xs]
z = np.round(z, 3)
print(z)
print(z.shape)
z = np.array(z)

ones = np.ones((1, 101))
m = np.matmul(z.reshape((101, 1)), ones)
m = m.T
print(m.shape)
print(m)
#plt.imshow(m)
#plt.show()
xx, yy = np.meshgrid(xs, xs)
"""
plt.pcolor(xx, yy, m, cmap = "gray")
plt.title("Soft Edge Model")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
"""
gmat = np.zeros(m.shape)
for i in range(101):
  for j in range(101):
    gmat[i, j] = G(xs[i], xs[j], 10)
"""
plt.pcolor(xx, yy, gmat, cmap = "gray")
plt.show()
"""
from scipy.signal import convolve2d
conv = convolve2d(m, gmat, mode = "same", boundary = "symm")
"""
plt.pcolor(xx, yy, conv, cmap = "gray")
plt.title("Soft Edge Convolution with Gaussian")
plt.xlabel("x")
plt.ylabel("y")
"""
#plt.show()


sobel_x = np.array([[-1, 0, 1]])
print(sobel_x.shape)
output = convolve2d(conv, sobel_x, boundary="symm")
plt.imshow(output, cmap = "gray")
print(output)
print(output.shape)
plt.show()

"""
x = [i for i in range(-200, 201)]
x = [i for i in range(-511, 5)]
y = x
z = np.zeros((401, 401))
for i in range(-200, 201):
  print(i)
  for j in range(-200, 201):
    z[i + 200, j + 200] = J(i, j, 1)

plt.imshow(z)
plt.show()
"""
"""
xx, yy = np.meshgrid(x, y)
for i in range(xx):
  for j in range(yy):
    z[i, j]
"""
"""
z = J(xx, yy, 1)
plt.imshow(z)
plt.show()
"""

xs = np.linspace(-5, 5, 101)
ys = np.linspace(-5, 5, 101)
taus = np.linspace(-5, 5, 101)

maxval = -100000
(maxx, maxt) = (-110, -110)
for x in xs:
  for t in taus:
    val = np.square(G(x, 0, np.sqrt(1 +  (t ** 2)))) 
    if val  > maxval:
      maxval = val
      (maxx, maxt) = (x, t)

print(maxx, maxt)
