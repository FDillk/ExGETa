import json
from math import sqrt
import sklearn
import joblib
import pickle
from enum import Enum
import numpy as np
import mlflow
import yaml
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import pandas

class Modelformat(Enum):
    SKLEARNSVC = 1
    SKLEARNSVR = 2
    SKLEARNLINSVC = 3
    SKLEARNLINSVR = 4
    SKLEARNNUSVC = 5
    SKLEARNNUSVR = 6
    SKLEARNDTC = 7


METAFILE = 'inputs/options.json'
global mlmodel

global modelformat, trainedmodel, loaded_model

global config
config = {}

#mlflow.set_experiment("my-experiment")

def eval_regression(loaded, dataset, metafile):
    print("NOT SUPPORTED YET: Regression Models")
    if(metafile['datasetmeta']['labelloc'] == "f"): # Labels are in first columns
        y = dataset.iloc[: , :metafile['datasetmeta']["n_labels"]].values
        X = dataset.iloc[: , metafile['datasetmeta']["n_labels"]:].values
    else:                                           # Labels are in last columns
        y = dataset.iloc[: , -metafile['datasetmeta']["n_labels"]:].values
        X = dataset.iloc[: , :-metafile['datasetmeta']["n_labels"]].values
    predictions = loaded.predict(X)

    
def eval_classification(loaded, dataset, metafile):
    if(metafile['datasetmeta']['labelloc'] == "f"): # Labels are in first columns
        y = dataset.iloc[: , :metafile['datasetmeta']["n_labels"]].values
        X = dataset.iloc[: , metafile['datasetmeta']["n_labels"]:].values
    else:                                           # Labels are in last columns
        y = dataset.iloc[: , -metafile['datasetmeta']["n_labels"]:].values
        X = dataset.iloc[: , :-metafile['datasetmeta']["n_labels"]].values

    if(metafile['datasetmeta']['scaling'] == 'scale'):
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        X = StandardScaler().fit_transform(X)
        
    X_test = X[metafile['datasetmeta']['traintest_cutoff']:, :]

    y_test = y[metafile['datasetmeta']['traintest_cutoff']:]

    predictions = loaded.predict(X_test)
    print("Predictions calculated..")
    conf_matrix = np.array(sklearn.metrics.confusion_matrix(y_test, predictions))
    loggingdata = dict()

    # Evaluate each class separately
    for i in range(len(conf_matrix)):
        all = np.sum(conf_matrix)
        posclass = np.sum(conf_matrix[:, i])
        posreal = np.sum(conf_matrix[i, :])
        negclass = all - posclass
        negreal = all - posreal
        tp = conf_matrix[i, i]
        fp = posclass - tp
        tn = all + tp - posclass - posreal
        fn = negclass - tn
        tpr = tp/posreal
        tnr = tn/negreal
        ppv = tp/posclass
        npv = tn/negclass
        fnr = 1 - tpr
        fpr = 1 - tnr
        fdr = 1 - ppv
        fomr = 1 - npv
        lrplus = tpr/fpr
        lrminus = fnr/tnr
        pt = sqrt(fpr)/(sqrt(tpr) + sqrt(fpr))
        ts = tp/(tp + fn + fp)


        if tp is not None: loggingdata['tp_class_' + str(i)] = tp
        if fp is not None: loggingdata['fp_class_' + str(i)] = fp
        if tn is not None: loggingdata['tn_class_' + str(i)] = tn
        if fn is not None: loggingdata['fn_class_' + str(i)] = fn
        if tpr is not None: loggingdata['tpr_class_' + str(i)] = tpr
        if tnr is not None: loggingdata['tnr_class_' + str(i)] = tnr
        if ppv is not None: loggingdata['ppv_class_' + str(i)] = ppv
        if npv is not None: loggingdata['npv_class_' + str(i)] = npv
        if fnr is not None: loggingdata['fnr_class_' + str(i)] = fnr
        if fpr is not None: loggingdata['fpr_class_' + str(i)] = fpr
        if fdr is not None: loggingdata['fdr_class_' + str(i)] = fdr
        if fomr is not None: loggingdata['for_class_' + str(i)] = fomr
        if fpr != 0: loggingdata['pos_likelihood_ratio_class_' + str(i)] = lrplus
        loggingdata['neg_likelihood_ratio_class_' + str(i)] = lrminus
        loggingdata['pt_class_' + str(i)] = pt
        loggingdata['ts_class_' + str(i)] = ts
        loggingdata['prevalence_class_' + str(i)] = posreal/(posreal/negreal)
        loggingdata['accuracy_class_' + str(i)] = (tp+tn)/all
        loggingdata['balanced_accuracy_class_' + str(i)] = (tpr + tnr)/2
        loggingdata['f1_score_class_' + str(i)] = 2*tp / (2*tp + fp + fn)
        loggingdata['matthews_corrcoef_class_' + str(i)] = (tp*tn - fp*fn)/(sqrt(posclass * posreal * negreal * negclass))
        loggingdata['fowlkes_mallows_index_class_' + str(i)] = sqrt(ppv*tpr)
        loggingdata['informedness_class_' + str(i)] = tpr-tnr-1
        loggingdata['markedness_class_' + str(i)] = ppv+npv-1
        if lrminus != 0: loggingdata['diagnostic_odds_ratio_class_' + str(i)] = lrplus/lrminus


    print("Loggings: ", str(loggingdata))
    # SKlearn scoring for classification:
    loggingdata['balanced_accuracy_score'] = sklearn.metrics.balanced_accuracy_score(y_test, predictions)
    loggingdata['matthews_corrcoef'] = sklearn.metrics.matthews_corrcoef(y_test, predictions)
    loggingdata['accuracy_score'] = sklearn.metrics.accuracy_score(y_test, predictions)
    loggingdata['hamming_loss'] = sklearn.metrics.hamming_loss(y_test, predictions)
    loggingdata['zero_one_loss'] = sklearn.metrics.zero_one_loss(y_test, predictions)

    # So far only binary classification is supported with these, as mlFlow does not allow logging array metrics
    if metafile['datasetmeta']['n_classes'] == 2:
        loggingdata['hinge_loss'] = sklearn.metrics.hinge_loss(y_test, predictions)
        loggingdata['roc_auc_score'] = sklearn.metrics.roc_auc_score(y_test, predictions)
        loggingdata['top_two_accuracy_score'] = sklearn.metrics.top_k_accuracy_score(y_test, predictions, k=2)
        loggingdata['jaccard_score'] = sklearn.metrics.jaccard_score(y_test, predictions)
        loggingdata['log_loss'] = sklearn.metrics.log_loss(y_test, predictions)
        #(precision, recall, f1_score, support) = sklearn.metrics.precision_recall_fscore_support(y_test, predictions, average=None)
        #if precision is not None: loggingdata['precision'] = precision
        #if recall is not None: loggingdata['recall'] = recall
        #if f1_score is not None: loggingdata['f1_score'] = f1_score
        #if support is not None: loggingdata['support'] = support
        
    print("Loggings: ", str(loggingdata))
    return loggingdata

def exec_eval_task(loaded, dataset, meta):

    print("Executing Basic Accuracy Evaluation Task...")
    if modelformat == Modelformat.SKLEARNLINSVC or modelformat == Modelformat.SKLEARNSVC or modelformat == Modelformat.SKLEARNNUSVC or modelformat == Modelformat.SKLEARNDTC:
        return eval_classification(loaded, dataset, meta)

    #elif modelformat == Modelformat.SKLEARNLINSVR or modelformat == Modelformat.SKLEARNSVR or modelformat == Modelformat.SKLEARNNUSVR:
    #    return eval_regression(loaded, dataset)


# Reads the input files

def input_processing():
    print("Processing Input Files")
    meta = meta_processing()
#    config = model_processing(meta["modelpath"])
    loaded = model_processing(meta)
    dataset = dataset_processing(meta)
    return loaded, dataset, meta

# Reads the model file
def model_processing(metafile):

    loader_module = metafile['modelmeta']['loader_module']

    if(loader_module == "mlflow.sklearn"):
        with open("inputs/MLmodel", 'r') as f:
            mlmodel = yaml.load(f, Loader=yaml.FullLoader) 

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
            # infos = process_LinearSVC(loaded_model)
        else:
            print('Unsupported sklearn model! modeltype: ' + modeltype)

        return loaded_model
    elif(loader_module == "sklearn.joblib"):
        loaded_model = joblib.load(open(metafile['modelmeta']['model_path'], 'rb'))
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
        elif modeltype == sklearn.tree._classes.DecisionTreeClassifier: 
            print('sklearn Decision Tree model detected!')
            modelformat = Modelformat.SKLEARNDTC
        
        return loaded_model
        
    elif(loader_module == "sklearn.pickle"):
        loaded_model = pickle.load(open(metafile['modelmeta']['model_path'], 'rb'))
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
        elif modeltype == sklearn.tree._classes.DecisionTreeClassifier: 
            print('sklearn Decision Tree model detected!')
            modelformat = Modelformat.SKLEARNDTC
        
        return loaded_model
    else:
        print("UNSUPPORTED MODEL")
    

    
# Reads the dataset file

def dataset_processing(metafile):
    print("Processing Dataset Input: " + metafile['datasetmeta']["data_path"])
    if(metafile['datasetmeta']["data_format"] == "CSV"):
        dataset = pandas.read_csv(metafile['datasetmeta']["data_path"], sep=metafile['datasetmeta']["data_separator"])
        return dataset

# Reads the meta file

def meta_processing():
    print("Processing Metadata Input...")
    f = open(METAFILE) 
    meta = json.load(f)
    f.close()
    return meta


def metricsaver(tolog, meta):
    print("Saving metrics to MLflow...")
    mlflow.set_tracking_uri(meta["mlflow_uri"])
    with mlflow.start_run(run_name="Basic Accuracy Evaluation"):
        mlflow.log_metrics(tolog)


if __name__ == "__main__":
    
    (loaded, dataset, meta) = input_processing()
    tolog = exec_eval_task(loaded, dataset, meta)
    print("All Tasks completed")
    metricsaver(tolog, meta)

