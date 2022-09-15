from math import log, sqrt
import sklearn
import mlflow
import csv
import pickle
import numpy as np
from sklearn.svm import LinearSVC, SVC
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
mnist = pandas.read_csv(".\mnist_full_nnormalized.csv") 
iono = pandas.read_csv('ionosphere.data', sep=",")
traintestcutoff = 60000

#print(mnist.head())

#print(iono)
#Y = mnist.iloc[:100 , :1].values
#X = mnist.iloc[:100 , 1:].values
Y = mnist.iloc[: , :1].values
X = mnist.iloc[: , 1:].values

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
clf = SVC(C=0.5, kernel='rbf').fit(X_train,y_train)


pkl_filename = "pickle_mnist_gen.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(clf, file)
print("Pickle saved")

# Save to file in the current working directory
joblib_file = "joblib_mnist_gen.pkl"
joblib.dump(clf, joblib_file)
print("Joblib saved")

print(clf.score(X_test, y_test))

sys.exit()

slack = abs(y_train-clf.decision_function(X_train))
#for s in slack:
#    print("%.5f" % s)
mistakes = abs(y_train-clf.predict(X_train))
#print(y_train)

alphas = np.abs(clf.dual_coef_[0])

vcdim = clf.n_features_in_ + 1
l = len(slack)

remp = (1/(1*l)) * sum(mistakes)

eta = 0.05

rabound = remp + sqrt((vcdim*(log((2*l)/vcdim)+1) - log(eta/4) )/l)

print("R emp: " + str(remp))
print("VC Dim: " + str(vcdim))
print("Ra Bound: " + str(rabound))

#print("Alphas: ")
#print(alphas)
d = 0
dmp = 0
dpm = 0
p = 1
R = 1
x = 0
n = len(slack)
#print(n)
count = 0
nplus = 0
nminus = 0

#print(clf.support_vectors_)
i = 0
for s in slack:
    if(count < len(alphas)):
        if(X_train[i] in clf.support_vectors_):
            if(p * alphas[count] * R + s >= 1):
                if(y_train[i] == 1):
                    dmp = dmp + 1
                if(y_train[i] == 0):
                    dpm = dpm + 1
                d = d + 1
            if(2 * alphas[count] * R + s >= 1):
                x = x + 1
            count = count + 1
        else:
            if(s >= 1):
                x = x + 1
                if(y_train[i] == 1):
                    dmp = dmp + 1
                if(y_train[i] == 0):
                    dpm = dpm + 1
                d = d + 1
                #print("HIT before all SV found: " + str(y_train[i]))
    else:
        if(s >= 1):
            x = x + 1
            if(y_train[i] == 1):
                dmp = dmp + 1
            if(y_train[i] == 0):
                dpm = dpm + 1
            d = d + 1
            #print("HIT after all SV found: " + str(y_train[i]))
    if(y_train[i] == 1):
        nplus = nplus + 1
    if(y_train[i] == -1):
        nminus = nminus + 1
    i = i + 1
    #print("%.5f" % s)
#print(slack)
estimator = d / n
estRec = 1 - (dmp/nplus)
estPrec = (nplus - dmp)/(nplus - dmp + dpm)
estFone = ((2*nplus) - (2*dmp))/((2*nplus) - dmp + dpm)
nmax = max(nplus, nminus)
maxvalpredictoracc = nmax/n

print("Test Score: ")
print(clf.score(X_test, y_test))
print("Train Score: ")
print(clf.score(X_train, y_train) )
print("d Value: ")
print(d)
print("dmp Value: ")
print(dmp)
print("dpm Value: ")
print(dpm)
print("n Value: ")
print(n)
print("n+ Value: ")
print(nplus)
print("Estimator: ")
print(1 - estimator)
print("Est Rec: ")
print(estRec)
print("Est Prec: ")
print(estPrec)
print("Est F1: ")
print(estFone)

score = clf.score(X_test, y_test)
predictions = clf.predict(X_test)
(precision, recall, f1_score, support) = sklearn.metrics.precision_recall_fscore_support(y_test, predictions, average='binary')

score = 0.7


print("Saving PDF....")
# save FPDF() class into a
# variable pdf
pdf = FPDF()
 
# Add a page
pdf.add_page()
 
# set style and size of font
# that you want in the pdf
pdf.set_font("Arial", size = 20, style="B")
 

pdf.cell(200, 30, txt = "Bounds and Measures Report",
         ln = 1, align = 'C')
 
pdf.set_font("Arial", size = 15, style="")
# create a cell
pdf.cell(200, 10, txt = "Vapnik Chervonenkis dimension: " + str(vcdim),
         ln = 1)

# create a cell
pdf.cell(200, 10, txt = "Estimated Recall: " + str(estRec),
         ln = 2)
# create a cell
pdf.cell(200, 10, txt = "Observed Recall: " + str(recall),
         ln = 3)
         
# create a cell
pdf.cell(200, 10, txt = "Estimated Precision: " + str(estPrec),
         ln = 4)
# create a cell
pdf.cell(200, 10, txt = "Observed Precision: " + str(precision),
         ln = 5)

# create a cell
pdf.cell(200, 10, txt = "Estimated F1 measure: " + str(estFone),
         ln = 6)
# create a cell
pdf.cell(200, 10, txt = "Observed F1 measure: " + str(f1_score),
         ln = 7)
         
# create a cell
pdf.cell(200, 10, txt = "Estimated Error: " + str(1 - estimator),
         ln = 8)
         
# create a cell
pdf.cell(200, 10, txt = "Test Error Bound: " + str(rabound),
         ln = 9)
 
# add another cell
pdf.cell(200, 10, txt = "Observed Test Error: " + str(1 - score),
         ln = 10)

# Divider
pdf.cell(200, 20, txt = "---------------------------------------------------------------------------------------------",
        ln = 11, align = 'C')


if(maxvalpredictoracc >= score):
    pdf.set_text_color(194,8,8)
    pdf.multi_cell(200, 10, txt = "Test Accuracy is less or equal to that of a naive most-frequent-class classifier!\nThis represents a failed classification model!\n ",
         align = 'C')
elif(score < 0.75):
    pdf.set_text_color(235, 156, 9)
    pdf.multi_cell(200, 10, txt = "Test Accuracy exceeds that of a naive most-frequent-class classifier, however is less than 75%.\nDepending on the data this may be good or bad.\n ",
         align = 'C')
else:
    pdf.set_text_color(8,194,8)
    pdf.multi_cell(200, 10, txt = "Test Accuracy exceeds that of a naive most-frequent-class classifier and is higher than 75%.\n ",
         align = 'C')


if((1-score) > rabound):
    pdf.set_text_color(194,8,8)
    pdf.multi_cell(200, 10, txt = "Test Error exceeds bounds! Probability for the bound holding is 95 percent.\nPlease check the model and dataset!",
         align = 'C')
else:
    pdf.set_text_color(8,194,8)
    pdf.multi_cell(200, 10, txt = "Test Error is within the bounds.\nProbability for the bound holding is 95 percent.",
         align = 'C')

pdf.multi_cell(200, 24, " ")

pdf.set_font("Arial", size = 8, style="B")
pdf.set_text_color(0,0,0)
pdf.multi_cell(200, 10, "References: ")
pdf.set_font("Arial", size = 8, style="")
pdf.multi_cell(200, 5, "Burges, Christopher J. C.: A Tutorial on Support Vector Machines for Pattern Recognition. Data Mining and Knowledge Discovery, 2(2):121-167, jun 1998.\n ")
pdf.multi_cell(200, 5, "Joachims, Thorsten: Learning to Classify Text Using Support Vector Machines: Methods, Theory and Algorithms. Kluwer Academic Publishers, Norwell, Massachusetts, USA, 2002.\n ")
# save the pdf with name .pdf
pdf.output("BaM.pdf")  

#for i in range(slack.length()):
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

# Save to file in the current working directory
#joblib_file = "joblib_modeltestsmall.pkl"
#joblib.dump(clf, joblib_file)
#print("Joblib saved")

#print(clf.predict(X[0].reshape(1, -1)))
# enable autologging
#mlflow.sklearn.autolog()

# prepare training data
#X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
#y = np.array([1, 1, 0, 1])
##
#data = np.array([ [0, 3, 4,	5], [1, 5, 7, 8],
#    [1, 1, 9, 7],
#    [1, 2, 10, 8],
#    [0, 5, 2, 2],
#    [0, 4, 3, 1],
#    [1, 10, 4, 10],
#    [0, 5, 7, 3], #f
#    [1, 4, 2, 10],
#    [0, 4, 5, 2],
#    [1, 9, 8, 2],
#    [0, 8, 1, 4],
#    [1, 3, 6, 9],
#    [1, 6, 3, 7],
#    [1, 9, 1, 9],
#    [1, 7, 4, 8],
#    [0, 6, 1, 5],
#    [1, 3, 9, 4],
#    [1, 8, 6, 3],
#    [0, 2, 1, 9],
#    [0, 5, 3, 4],
#    [0, 1, 4, 8],
#    [0, 7, 2, 2],
#    [0, 3, 6, 1],
#    [0, 5, 3, 4],
#    [0, 1, 1, 1],
#    [0, 1, 1, 1]#
#])
#with open('example.csv', 'w', newline ='') as f: 
#    write = csv.writer(f) 
#    write.writerows(data) 
#
#X = np.array(data[:, 1:])
#y = np.array(data[:, 0])
#
#print(str(X))
#print(str(y))
#print(str(dat#a[:]))

#x, y = make_classification(n_samples=5000, n_features=10, 
#                           n_classes=3, 
#                           n_clusters_per_class=1)



#xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15)

#svc = SVC(verbose=0)
#print(svc)

#SVC(C=1.0, class_weight=None, random_state=None, tol=0.0001,
#          verbose=0)

#with mlflow.start_run() as run:
#    lsvc.fit(xtrain, ytrain)
#svc.fit(xtrain, ytrain)
#score = svc.score(xtest, ytest)
#print("Xtrain: " + str(xtrain))
#print("Ytrain: " + str(ytrain))
#print("Xtest: " + str(xtest))
#print("Ytest: " + str(ytest))

#print("Score: ", score)
#print("Pred: " + str(svc.predict(xtest)))

#mlflow.sklearn.save_model(svc, modelpath)
#print("Model saved")

#sk_model = mlflow.sklearn.load_model(modelpath)
#score = sk_model.score(X, y)
#print("Score 2: ", sklearn.metrics.confusion_matrix(y, sk_model.predict(X)))
#print("Score 2: ", np.delete(sklearn.metrics.confusion_matrix(y, sk_model.predict(X)), 0, 0))

#print(sk_model.predict(X))

