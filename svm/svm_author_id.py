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

### print result for element 10
print "Label of element 10: ", labels_pred[10]

### print result for element 26
print "Label of element 26: ", labels_pred[26]

### print result for element 50
print "Label of element 50: ", labels_pred[50]

### print number of classifications for chris and Sara
print "Number of predicted emails by sara: ", (labels_pred == 0).sum()
print "Number of predicted emails by chris: ", (labels_pred == 1).sum()

#########################################################


