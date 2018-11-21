import numpy as np
from matplotlib import pyplot as pl

Training_Data = np.loadtxt("mnist/MNIST-Train-cropped.txt")
Training_Labels = np.loadtxt("mnist/MNIST-Train-Labels-cropped.txt")
Test_Data = np.loadtxt("mnist/MNIST-Test-cropped.txt")
Test_Labels = np.loadtxt("mnist/MNIST-Test-Labels-cropped.txt")

train_data = Training_Data.reshape(10000, 784)
test_image = train_data[0].reshape(28, 28).T
test_data = Test_Data.reshape(2000, 784)


#pl.imshow(test_image)
#pl.show()

train_labels = Training_Labels

#Must figure out data format convention
def KNN(train_data, train_labels, x, k):
  X = np.tile(x, (train_data.shape[0], 1)).T
  Xt = np.transpose(train_data)
  norms = np.linalg.norm(X - Xt, axis = 0)
  indices = np.argsort(norms)[0:k]
  labels = [train_labels[i] for i in indices]
  result =  np.argmax(np.bincount(labels))
  print(result)
  return result


Ks = [i for i in range(0, 34) if i % 2 == 1]


def validate(test_results, real_labels):
  print(len(test_results))
  print(len(real_labels))
  count = 0
  for i in range(len(test_results)):
    if test_results[i] == real_labels[i]:
        count += 1
  print(count)
  return float(count) / len(real_labels)


def testResults(k, labels):
  res = [KNN(train_data, train_labels, val, k) for val in test_data]
  val = validate(res, labels)
  print("Printing results")
  print("Printing results")
  print("Printing results")
  print("Printing results")
  print("Printing results")
  print("Printing results")
  print(val)
  print(val)
  print(val)
  print(val)
  print(val)
  print(val)
  print("Printed Result")
  print("Printed Result")
  print("Printed Result")
  print("Printed Result")
  print("Printed Result")
  print("Printed Result")
  return val

#print(validate(results, Test_Labels))


def run():
  return [testResults(k, Test_Labels) for k in Ks] 

print(run())

