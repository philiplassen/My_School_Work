import numpy as np

train = np.loadtxt("parkinsonsTrainStatML.dt")
test = np.loadtxt("parkinsonsTestStatMl.dt")

train_data = train[:, 0:21]
test_data = test[:, 0:21]

train_labels = train[:, 22]
test_labels = test[:, 22]
print("The Training Means are : ")
print(np.mean(train_data, axis = 0))
print("")
print("The Training Variances are : ")
print(np.var(train_data, axis = 0))



print("The testing Means are : ")
print(np.mean(test_data, axis = 1))
print("")
print("The testing Variances are : ")
print(np.var(test_data, axis = 1))


temp = np.transpose(np.subtract(np.transpose(train_data), np.mean(train_data, axis = 1)))
train_n = np.transpose(np.divide(np.transpose(temp), np.std(temp, axis = 1)))
print(np.mean(train_n, axis = 1))
print(np.var(train_n, axis = 1))



temp = np.transpose(np.subtract(np.transpose(test_data), np.mean(test_data, axis = 1)))
test_n = np.transpose(np.divide(np.transpose(temp), np.std(temp, axis = 1)))
print(np.mean(test_n, axis = 1))
print(np.var(test_n, axis = 1))


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


