from math import log, sqrt
import sklearn
import mlflow
import csv
import pickle
import numpy as np
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
import joblib
import pandas
from codecarbon import EmissionsTracker
from fpdf import FPDF
import sys
from sklearn import tree
import graphviz 
from sklearn.tree import export_text

modelpath = "/modelsave/model2"
traintestcutoff = 100
iris = pandas.read_csv("shuffled.csv", sep=',')
#ds = iris.sample(frac=1)
#ds.to_csv('shuffled.csv', index=False)

Y = iris.iloc[: , -1:].values
X = iris.iloc[: , :-1].values

label_encoder = LabelEncoder()
#Y = label_encoder.fit_transform(Y)
#X = StandardScaler().fit_transform(X)

X_train = X[:traintestcutoff, :]
X_test = X[traintestcutoff:, :]

y_train = Y[:traintestcutoff]
y_test = Y[traintestcutoff:]

clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(X_train, y_train)

r = export_text(clf) #, feature_names=iris['feature_names'])
print(r)

print(clf.score(X_test, y_test))


pkl_filename = "pickle_iris.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)
print("Pickle saved")

# Save to file in the current working directory
joblib_file = "joblib_iris.pkl"
joblib.dump(clf, joblib_file)
print("Joblib saved")
