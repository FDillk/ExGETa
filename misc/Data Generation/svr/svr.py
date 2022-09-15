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

modelpath = "/modelsave/model2"
cali = pandas.read_csv('calihousing_imp.csv', sep=",")
traintestcutoff = 16346

#print(mnist.head())

#print(iono)
#Y = mnist.iloc[:100 , :1].values
#X = mnist.iloc[:100 , 1:].values
Y = cali.iloc[: , -1:].values
X = cali.iloc[: , :-1].values

#Y = iono.iloc[:, 34:].values
#X = iono.iloc[:, :34].values

#print(X)
#print(Y)

label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(Y)
#X = preprocessing.normalize(StandardScaler().fit_transform(X))
X = StandardScaler().fit_transform(X)

X_train = X[:traintestcutoff, :]
X_test = X[traintestcutoff:, :]

y_train = Y[:traintestcutoff]
y_test = Y[traintestcutoff:]

print(X.var())

#X_train , X_test, y_train, y_test = train_test_split(X,Y)

#print(y_train)

#tracker = EmissionsTracker(output_dir="./")
#tracker.start()
# GPU intensive training code
reg = SVR(kernel='rbf').fit(X_train,y_train)
#clf = SVC(C=0.5, kernel='rbf').fit(X_train,y_train)


pkl_filename = "pickle_calihousing.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(reg, file)
print("Pickle saved")

# Save to file in the current working directory
joblib_file = "joblib_calihousing.pkl"
joblib.dump(reg, joblib_file)
print("Joblib saved")

print(reg.score(X_test, y_test))

sys.exit()
#    print(slack)


#emissions = tracker.stop()
#print(emissions)

#score = clf.score(X_test, y_test)
#print("Score: ", score)

#csvfile = open('emissions.csv', newline='')#
#reader = csv.DictReader(csvfile)
#duration = 0
#for row in reader:
#    duration = row['duration']
#    
#print(type(float(duration)))
#print(float(duration))

#mlflow.sklearn.save_model(clf, modelpath)
#print("MLflow saved")

pkl_filename = "pickle_ionobound.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)
print("Pickle saved")
