import json
from math import log, sqrt
import sklearn
import joblib
import yaml
import pickle
from enum import Enum
import numpy as np
import mlflow
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import preprocessing
import joblib
import pandas
from fpdf import FPDF

class Modelformat(Enum):
    SKLEARNSVC = 1
    SKLEARNSVR = 2
    SKLEARNLINSVC = 3
    SKLEARNLINSVR = 4
    SKLEARNNUSVC = 5
    SKLEARNNUSVR = 6


METAFILE = 'inputs/options.json'
#MODELPATH = "inputs"
global mlmodel

global modelformat, trainedmodel, loaded_model

global config
config = {}

#mlflow.set_experiment("my-experiment")


def generatePDFReport(vcdim, estRec, recall, estPrec, precision, estFone, f1_score, estimator, rabound, score, maxvalpredictoracc):
        
    print("Saving PDF....")
    # save FPDF() class into a
    # variable pdf
    pdf = FPDF()
    
    # Add a page
    pdf.add_page()
    
    # set style and size of headline font
    pdf.set_font("Arial", size = 20, style="B")
    

    pdf.cell(200, 30, txt = "Bounds and Measures Report",
            ln = 1, align = 'C')
    
    # set style and size of text font
    pdf.set_font("Arial", size = 15, style="")
    
    pdf.cell(200, 10, txt = "Vapnik Chervonenkis dimension: " + str(vcdim),
            ln = 1)

    pdf.cell(200, 10, txt = "Estimated Recall: " + str(estRec),
            ln = 2)
            
    pdf.cell(200, 10, txt = "Observed Recall: " + str(recall),
            ln = 3)
            
    pdf.cell(200, 10, txt = "Estimated Precision: " + str(estPrec),
            ln = 4)

    pdf.cell(200, 10, txt = "Observed Precision: " + str(precision),
            ln = 5)

    pdf.cell(200, 10, txt = "Estimated F1 measure: " + str(estFone),
            ln = 6)
            
    pdf.cell(200, 10, txt = "Observed F1 measure: " + str(f1_score),
            ln = 7)
            
    pdf.cell(200, 10, txt = "Estimated Error: " + str(1 - estimator),
            ln = 8)
            
    pdf.cell(200, 10, txt = "Test Error Bound: " + str(rabound),
            ln = 9)
    
    pdf.cell(200, 10, txt = "Observed Test Error: " + str(1 - score),
            ln = 10)

    # Divider
    pdf.cell(200, 20, txt = "---------------------------------------------------------------------------------------------",
            ln = 11, align = 'C')

    # Colored generated Sentences
    if(maxvalpredictoracc >= score):
        pdf.set_text_color(194,8,8) # set textcolor to red
        pdf.multi_cell(200, 10, txt = "Test Accuracy is less or equal to that of a naive most-frequent-class classifier!\nThis represents a failed classification model!\n ",
            align = 'C')
    elif(score < 0.75):
        pdf.set_text_color(235, 156, 9) # set textcolor to orange
        pdf.multi_cell(200, 10, txt = "Test Accuracy exceeds that of a naive most-frequent-class classifier, however is less than 75%.\nDepending on the data this may be good or bad.\n ",
            align = 'C')
    else:
        pdf.set_text_color(8,194,8) # set textcolor to green
        pdf.multi_cell(200, 10, txt = "Test Accuracy exceeds that of a naive most-frequent-class classifier and is higher than 75%.\n ",
            align = 'C')


    if((1-score) > rabound):
        pdf.set_text_color(194,8,8) # set textcolor to red
        pdf.multi_cell(200, 10, txt = "Test Error exceeds bounds! Probability for the bound holding is 95 percent.\nPlease check the model and dataset!",
            align = 'C')
    else:
        pdf.set_text_color(8,194,8) # set textcolor to green
        pdf.multi_cell(200, 10, txt = "Test Error is within the bounds.\nProbability for the bound holding is 95 percent.",
            align = 'C')

    pdf.multi_cell(200, 24, " ")

    pdf.set_font("Arial", size = 8, style="B")
    pdf.set_text_color(0,0,0)
    # Insert references for the Bounds
    pdf.multi_cell(200, 10, "References: ")
    pdf.set_font("Arial", size = 8, style="")
    pdf.multi_cell(200, 5, "Burges, Christopher J. C.: A Tutorial on Support Vector Machines for Pattern Recognition. Data Mining and Knowledge Discovery, 2(2):121-167, jun 1998.\n ")
    pdf.multi_cell(200, 5, "Joachims, Thorsten: Learning to Classify Text Using Support Vector Machines: Methods, Theory and Algorithms. Kluwer Academic Publishers, Norwell, Massachusetts, USA, 2002.\n ")
    # save the pdf with name Boundsreport.pdf
    pdf.output("Boundsreport.pdf")  
    
def eval_bounds(loaded, dataset, metafile):

    if(metafile['datasetmeta']['labelloc'] == "f"): # Labels are in first columns
        Y = dataset.iloc[: , :metafile['datasetmeta']["n_labels"]].values
        X = dataset.iloc[: , metafile['datasetmeta']["n_labels"]:].values
    else:                                           # Labels are in last columns
        Y = dataset.iloc[: , -metafile['datasetmeta']["n_labels"]:].values
        X = dataset.iloc[: , :-metafile['datasetmeta']["n_labels"]].values

    traintestcutoff = metafile['datasetmeta']['traintest_cutoff']

    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)
    X = preprocessing.normalize(StandardScaler().fit_transform(X))

    X_train = X[:traintestcutoff, :]
    X_test = X[traintestcutoff:, :]

    y_train = Y[:traintestcutoff]
    y_test = Y[traintestcutoff:]

        
    clf = loaded #SVC(C=0.5, kernel='linear').fit(X_train,y_train)

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

    d = 0
    dmp = 0
    dpm = 0
    p = 1
    R = 1
    x = 0
    n = len(slack)
    count = 0
    nplus = 0
    nminus = 0

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
        else:
            if(s >= 1):
                x = x + 1
                if(y_train[i] == 1):
                    dmp = dmp + 1
                if(y_train[i] == 0):
                    dpm = dpm + 1
                d = d + 1
        if(y_train[i] == 1):
            nplus = nplus + 1
        if(y_train[i] == -1):
            nminus = nminus + 1
        i = i + 1
        
    estimator = d / n
    estRec = 1 - (dmp/nplus)
    estPrec = (nplus - dmp)/(nplus - dmp + dpm)
    estFone = ((2*nplus) - (2*dmp))/((2*nplus) - dmp + dpm)
    nmax = max(nplus, nminus)
    #Value for always predicting most common class
    maxvalpredictoracc = nmax/n 

        
    score = clf.score(X_test, y_test)
    predictions = clf.predict(X_test)
    (precision, recall, f1_score, support) = sklearn.metrics.precision_recall_fscore_support(y_test, predictions, average='binary')

    generatePDFReport(vcdim, estRec, recall, estPrec, precision, estFone, f1_score, estimator, rabound, score, maxvalpredictoracc)

    loggingdata = dict()
    loggingdata['Evaluated Score'] = score
    loggingdata['Estimated Recall'] = estRec
    loggingdata['Evaluated Recall'] = recall
    loggingdata['Estimated Precision'] = estPrec
    loggingdata['Evaluated Precision'] = precision
    loggingdata['Estimated F1'] = estFone
    loggingdata['Evaluated F1'] = f1_score
    loggingdata['Estimated Recall'] = estRec
    loggingdata['Emprirical Error'] = remp
    loggingdata['Error Bound - Joachims'] = rabound
    loggingdata['Vapnik Chervonenkis Dimension'] = vcdim
#    print("Loggings: ", str(loggingdata))
    return loggingdata

def exec_eval_task(loaded, dataset, metafile):

    print("Executing Bounds Report Evaluation Task...")
    if modelformat == Modelformat.SKLEARNLINSVC or modelformat == Modelformat.SKLEARNSVC or modelformat == Modelformat.SKLEARNNUSVC:
        return eval_bounds(loaded, dataset, metafile)

# Reads the input files

def input_processing():
    print("Processing Input Files")
    meta = meta_processing()
#    config = model_processing(meta["modelpath"])
    loaded = model_processing(meta)
    dataset = dataset_processing(meta)
    return loaded, dataset, meta

# Reads the model file
def model_processing(modelfile):

    #trainedmodel = joblib.load(modelfile)
    #print("Processing Model Input: " + str(type(trainedmodel)))
    #modeltype = type(trainedmodel)
    loader_module = modelfile['modelmeta']['loader_module']

    if(loader_module == "mlflow.sklearn"):
        with open("inputs/MLmodel", 'r') as f:
            mlmodel = yaml.load(f, Loader=yaml.FullLoader) # DOPPELT

        modelname = mlmodel['flavors']['sklearn']['pickled_model']
        loaded_model = pickle.load(open("inputs/" + modelname, 'rb'))
        modeltype = type(loaded_model)
        global modelformat

        # sklearn SVC Modell
        if modeltype == sklearn.svm._classes.SVC: 
            print('sklearn SVC model detected!')
            modelformat = Modelformat.SKLEARNSVC
        elif modeltype == sklearn.svm._classes.NuSVC: 
            print('sklearn NuSVC model detected!')
            modelformat = Modelformat.SKLEARNNUSVC
        elif modeltype == sklearn.svm._classes.LinearSVC: 
            print('sklearn LinearSVC model detected!')
            modelformat = Modelformat.SKLEARNLINSVC
        else:
            print('Unsupported sklearn model! modeltype: ' + modeltype)

        return loaded_model
    elif(loader_module == "sklearn.joblib"):
        loaded_model = joblib.load(open(modelfile['modelmeta']['model_path'], 'rb'))
        modeltype = type(loaded_model)
        # sklearn SVC Modell
        if modeltype == sklearn.svm._classes.SVC: 
            print('sklearn SVC model detected!')
            modelformat = Modelformat.SKLEARNSVC
        elif modeltype == sklearn.svm._classes.NuSVC: 
            print('sklearn NuSVC model detected!')
            modelformat = Modelformat.SKLEARNNUSVC
        elif modeltype == sklearn.svm._classes.LinearSVC: 
            print('sklearn LinearSVC model detected!')
            modelformat = Modelformat.SKLEARNLINSVC
        
        return loaded_model
    elif(loader_module == "sklearn.pickle"):
        loaded_model = pickle.load(open(modelfile['modelmeta']['model_path'], 'rb'))
        modeltype = type(loaded_model)
        # sklearn SVC Modell
        if modeltype == sklearn.svm._classes.SVC: 
            print('sklearn SVC model detected!')
            modelformat = Modelformat.SKLEARNSVC
        elif modeltype == sklearn.svm._classes.NuSVC: 
            print('sklearn NuSVC model detected!')
            modelformat = Modelformat.SKLEARNNUSVC
        elif modeltype == sklearn.svm._classes.LinearSVC: 
            print('sklearn LinearSVC model detected!')
            modelformat = Modelformat.SKLEARNLINSVC
        
        return loaded_model
    else:
        print("UNSUPPORTED MODEL")
    

    
# Reads the dataset file

def dataset_processing(meta):
    print("Processing Dataset Input...")
    if(meta['datasetmeta']["data_format"] == "CSV"):
        dataset = pandas.read_csv(meta['datasetmeta']["data_path"], sep=meta['datasetmeta']["data_separator"])
    else:
        # iono = pandas.read_csv(datafile, sep=",")
        dataset = 0

    return dataset

# Reads the meta file

def meta_processing():
    f = open(METAFILE) 
    meta = json.load(f)
    f.close()
    print("Processing Metadata Input: ") # + str(meta))

    return meta


def metricsaver(tolog, meta):
    print("Saving metrics to MLflow...")
    mlflow.set_tracking_uri(meta["mlflow_uri"])
    with mlflow.start_run(run_name="Bounds Report Evaluation"):
        mlflow.log_metrics(tolog)
        mlflow.log_artifact("Boundsreport.pdf")


if __name__ == "__main__":

    (loaded, dataset, meta) = input_processing()
    tolog = exec_eval_task(loaded, dataset, meta)
    print("Task completed")
    metricsaver(tolog, meta)
