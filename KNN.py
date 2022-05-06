import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.datasets import load_iris 

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print ('Number of classes: %d' %len(np.unique(iris_y)))
print ('Number of data points: %d' %len(iris_y))


X0 = iris_X[iris_y == 0,:]
print ('\nSamples from class 0:\n', X0[:5,:])

X1 = iris_X[iris_y == 1,:]
print ('\nSamples from class 1:\n', X1[:5,:])

X2 = iris_X[iris_y == 2,:]
print ('\nSamples from class 2:\n', X2[:5,:])
from sklearn.model_selection import train_test_split
# randomly choose test set and trainning set from dataset
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)

print ("Training size: %d" %len(y_train))
print ("Test size    : %d" %len(y_test))
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
# K = 1, chi xet 1 diem trainning data gan nhat va lay label cua dem do du doan cho diem test
clf.fit(X_train, y_train)
# 学習用のデータと結果を学習する,fit()
y_pred = clf.predict(X_test)

print ("Print results for 20 test data points:")
print ("Predicted labels: ", y_pred[20:40]) # slicing array x[start:end] include start, not include end
print ("Ground truth    : ", y_test[20:40])
from sklearn.metrics import accuracy_score
print ("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))