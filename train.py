from sklearn import svm
import numpy as np
from sklearn.externals import joblib

train_data = np.loadtxt("train_data.txt")

#Number of features
n = train_data.shape[1] - 1

#Extract the features 
X = train_data[:, 0:n]
#print X
#Extract the labels
y = train_data[:, n:].transpose()[0]
#print y

clf = svm.SVC()
clf.fit(X, y)

#Save the trained model
joblib.dump(clf, 'model.pkl') 

print "Saved model file: model.pkl"