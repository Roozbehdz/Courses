from assignment1.cs231n.classifiers.k_nearest_neighbor import KNearestNeighbor
import random
import numpy as np
from assignment1.cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
from cross_validation import cross_validation


# Load the raw CIFAR-10 data.
cifar10_dir = 'assignment1/cs231n/datasets/cifar-10-batches-py'
X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

# # Subsample the data for more efficient code execution in this exercise
# num_training = 10000
# mask = list(range(num_training))
# X_train = X_train[mask]
# y_train = y_train[mask]

num_test = X_test.shape[0]
# mask = list(range(num_test))
# X_test = X_test[mask]
# y_test = y_test[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
print(X_train.shape, X_test.shape)


# Create a kNN classifier instance.
# Remember that training a kNN classifier is a noop:
# the Classifier simply remembers the data and does no further processing
classifier = KNearestNeighbor()
classifier.train(X_train, y_train)

#Cross-validation
# validations = cross_validation(X_train,y_train)
# print(validations)


# Based on the cross-validation results above, choose the best value for k,
# retrain the classifier using all the training data, and test it on the test
# data. You should be able to get above 28% accuracy on the test data.


#Evaluation
y_test_pred = classifier.predict(X_test, k=100)

# Compute and display the accuracy
num_correct = np.sum(y_test_pred == y_test)
accuracy = float(num_correct) / num_test
print('Got %d / %d correct => accuracy: %f' %
      (num_correct, num_test, accuracy))
