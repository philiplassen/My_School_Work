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

"""
sobel_x = np.array([[-1, 0, 1]])
print(sobel_x.shape)
output = convolve2d(conv, sobel_x, boundary="symm")
plt.imshow(output, cmap = "gray")
print(output)
print(output.shape)
plt.show()
"""
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



from skimage import color, io
im = np.array(io.imread('hand.tiff', plugin='pil')).astype(float)
print(im.shape)
print(im)
"""plt.imshow(im)
plt.show()"""


import functions as fct

tau = 5
n = 100
img_flower = im
H_map = fct.H(img_flower, tau)
fig, ax = plt.subplots(2, 1, figsize=(4, 6))
ax[0].imshow(img_flower, cmap='gray')
print(H_map)
ax[1].imshow(H_map.astype("float"), cmap='gray')
image_path = "Im/"
fig.savefig(image_path + '2_blob_detection_t5_H.png')


max_val, min_val = fct.find_max(H_map, n = n)
fig, ax = plt.subplots()
plt.imshow(img_flower, cmap = 'gray')
plt.scatter(max_val[:, 1], max_val[:, 0], marker = 'o', s = .1,  color = 'blue', alpha =1) #marker = 'x', '1'
for i in range(n):
  circle_max = plt.Circle((max_val[i, 1], max_val[i, 0]), 2*tau, color='b', fill = False)
  ax.add_artist(circle_max)
plt.scatter(min_val[:, 1], min_val[:, 0], marker = 'o', s = .1, color = 'red', alpha = 1) #marker = 'x', '1'
for i in range(n):
  circle_min = plt.Circle((min_val[i, 1], min_val[i, 0]), 2*tau, color='r', fill = False)
  ax.add_artist(circle_min)

fig.savefig(image_path + '2_blob_detection_t5_blob.png')
tau_arr = [3, 5, 7, 10, 15, 20, 25, 50]
#tau_arr = [1, 2,1, 3]
n = 100
fig, ax = plt.subplots(figsize = (12, 9))
ax.imshow(img_flower, cmap = 'gray')
for i in range(len(tau_arr)):
  H_map_test = fct.H(img_flower, tau_arr[i])
  max_val_test, min_val_test = fct.find_max(H_map_test, n = n)
  ax.scatter(max_val_test[:, 1], max_val_test[:, 0], marker = '.', s = .1, color = 'blue', alpha =1) #marker = 'x', '1'
  for j in range(n):
    circle_max = plt.Circle((max_val_test[j, 1], max_val_test[j, 0]), 1.8*tau_arr[i], color='b', fill = False)
    ax.add_artist(circle_max)

  ax.scatter(min_val_test[:, 1], min_val_test[:, 0], marker = '.', s = .1, color = 'red', alpha = 1) #marker = 'x', '1'
  for j in range(n):
    circle_min = plt.Circle((min_val_test[j, 1], min_val_test[j, 0]), 1.8*tau_arr[i], color='r', fill = False)
    ax.add_artist(circle_min)

fig.savefig(image_path + '2_blob_detection_test_20compact.png')






tau_arr = [3, 5, 7, 10, 12, 15, 20, 25, 30]
#tau_arr = [1, 2, 3]
H_map_test, ind_tau = fct.H_multitau(img_flower, tau_arr)
n = 100  #max = 421500
max_val_test, min_val_test = fct.find_local_max(H_map_test, n = n, k=1)

fig, ax = plt.subplots(figsize = (12, 9))
ax.imshow(img_flower, cmap = 'gray')
ax.scatter(max_val_test[:, 1], max_val_test[:, 0], marker = '.', s = .1, color = 'blue', alpha =1) #marker = 'x', '1'
for j in range(n):
  circle_max = plt.Circle((max_val_test[j, 1], max_val_test[j, 0]), 1.8*tau_arr[ind_tau[int(max_val_test[j, 0]), int(max_val_test[j, 1])]], color='b', fill = False)
  ax.add_artist(circle_max)

ax.scatter(min_val_test[:, 1], min_val_test[:, 0], marker = '.', s = .1, color = 'red', alpha = 1) #marker = 'x', '1'
for j in range(n):
  circle_min = plt.Circle((min_val_test[j, 1], min_val_test[j, 0]), 1.8*tau_arr[ind_tau[int(max_val_test[j, 0]), int(max_val_test[j, 1])]], color='r', fill = False)
  ax.add_artist(circle_min)

fig.savefig(image_path + '2_blob_detection_test_20compact_multi.png')

conv = convolve2d(m, gmat, mode = "same", boundary = "symm")

