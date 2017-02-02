from sklearn import svm
import numpy as np
from sklearn.externals import joblib

#Load the trained model
clf = joblib.load('model.pkl') 

test_data = np.loadtxt("test_data.txt")

#Number of features
n = test_data.shape[1] - 1
#Number of test data samples
m = test_data.shape[0]

#Extract the features 
X = test_data[:, 0:n]

#Extract the labels
y = test_data[:, n:].transpose()[0]

p = clf.predict(X)

correctPredictions = np.sum(p==y)
accuracy = (correctPredictions * 100) / m

print "Prediction accuracy: %s%%" % accuracy
