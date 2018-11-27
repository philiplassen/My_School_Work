import numpy as np
from matplotlib import pyplot as pl

Training_Data = np.loadtxt("mnist/MNIST-Train-cropped.txt")
Training_Labels = np.loadtxt("mnist/MNIST-Train-Labels-cropped.txt")
Test_Data = np.loadtxt("mnist/MNIST-Test-cropped.txt")
Test_Labels = np.loadtxt("mnist/MNIST-Test-Labels-cropped.txt")

train_data = Training_Data.reshape(10000, 784)
test_image = train_data[0].reshape(28, 28).T
test_data = Test_Data.reshape(2000, 784)
train_labels = Training_Labels
test_labels = Test_Labels

ctrain_data = train_data[0:8000, :]
ctest_data = train_data[8000:10000, :]
ctrain_labels = train_labels[0:8000]
ctest_labels = train_labels[8000:10000]




def genImages(num, data, labels):
  results = [data[i, :] for i in range(np.shape(data)[0]) if labels[i] == num]
  return np.array(results)


Ks = [i for i in range(0, 34) if i % 2 == 1]

def knnCompare(v1, v2, input_train_data, input_test_data, input_train_labels, input_test_labels):
  #cross validation part  
  print("Generating data for the cross validation")
  tr1 = genImages(v1, input_train_data, input_train_labels)
  tr2 = genImages(v2, input_train_data, input_train_labels)
  trData = np.vstack((tr1, tr2))
  te1 = genImages(v1, input_test_data, input_test_labels)
  te2 = genImages(v2, input_test_data, input_test_labels)
  teData = np.vstack((te1, te2))
  print("Generating results for the cross validation")
  tr1Labels = np.empty(tr1.shape[0])
  tr1Labels.fill(v1)
  tr2Labels = np.empty(tr2.shape[0])
  tr2Labels.fill(v2)
  print(tr1Labels.shape)
  print(tr2Labels.shape)
  trLabels = np.append(tr1Labels, tr2Labels)
  te1Labels = np.empty(te1.shape[0])
  te1Labels.fill(v1)
  te2Labels = np.empty(te2.shape[0])
  te2Labels.fill(v2)
  teLabels = np.append(te1Labels, te2Labels)
  print("Running through the Ks")
  res = [KNN(trData, trLabels, dat) for  dat in teData]
  return validate(res, teLabels)



#Must figure out data format convention
def KNN(train_data, train_labels, x):
  X = np.tile(x, (train_data.shape[0], 1)).T
  Xt = np.transpose(train_data)
  norms = np.linalg.norm(X - Xt, axis = 0)
  result = []
  for k in Ks:
    indices = np.argsort(norms)[0:k]
    labels = [train_labels[i] for i in indices]
    result +=  [np.argmax(np.bincount(labels))]
  return result


Ks = [i for i in range(0, 34) if i % 2 == 1]


def validate(test_results, real_labels):
  print(len(test_results))
  print(len(real_labels))
  test_res = []
  for j in range(len(Ks)):
    count = 0
    for i in range(len(test_results)):
      if test_results[i][j] == real_labels[i]:
          count += 1
    test_res += [1 - (float(count) / len(real_labels))]
  return test_res


def genPlot(v1, v2):
  pl.xlabel("k")
  pl.ylabel("Error")
  pl.title("Error for Digit Classification of " + str(v1) + " and " + str(v2))
  train_err = knnCompare(v1, v2, ctrain_data, ctest_data, ctrain_labels, ctest_labels)
  test_err = knnCompare(v1, v2, train_data, test_data, train_labels, test_labels)
  print("Printing Training Error")
  print(train_err)
  print("Printing Test Error")
  print(test_err)
  pl.plot(Ks, train_err, marker = 'o', label="Training Error")
  pl.plot(Ks, test_err, marker = 'o', label="Test Error")
  pl.legend(["Training Error", "Test Error"])
  pl.show()
