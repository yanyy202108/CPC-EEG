import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import scipy.io as scio
import sys

data1 = scio.loadmat(r'D:\contrastive-predictive-coding-master\result_xyzhuan11.mat')
n_samples, n_features = data1['matrix_1'].shape
print(sys.getsizeof(n_samples), n_features)

max_test = []
test = []
train = []
for i in range(10):

    train_data, test_data,train_label, test_label = train_test_split(data1['matrix_1'],data1['matrix_2'], random_state=6, train_size=0.6, test_size=0.4)
    #train_label, test_label = train_test_split(data1['matrix_2'], random_state=42, train_size=0.75, test_size=0.25)#9 = 57 19=62


classifier = svm.SVC(C=400, kernel='rbf', gamma='auto', decision_function_shape='ovo')
classifier.fit(train_data, train_label.ravel())

pre_train = classifier.predict(train_data)
pre_test = classifier.predict(test_data)
train.append(accuracy_score(train_label, pre_train))
test.append(accuracy_score(test_label, pre_test))
# print(pre_test)
print(test)
print(max(test))
max_index = test.index(max(test))

print("train:", train[max_index])
print("test:", test[max_index])
