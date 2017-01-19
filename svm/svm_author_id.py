#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

### create the SVC classifier and train it.
from sklearn import svm
clf = svm.SVC(kernel='rbf', C=10000.0)
timestampBeforeFitting = time()
clf.fit(features_train, labels_train)
print "trainig time: ", round(time()-timestampBeforeFitting,3), "s"

### predict the labels for the feature test set
timestampBeforePrediction = time()
labels_pred = clf.predict(features_test)
print "prediction time: ", round(time() - timestampBeforePrediction,3), "s"

### print the accuracy of the predicted labels.
from sklearn.metrics import accuracy_score
print "Accuracy: " + str(accuracy_score(labels_test, labels_pred))
#########################################################


