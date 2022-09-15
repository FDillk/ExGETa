import argparse
import json
from math import sqrt
import sklearn
import joblib
import yaml
import csv
import pickle
from enum import Enum
from sklearn.model_selection import train_test_split
import numpy as np

class Modelformat(Enum):
    SKLEARNSVC = 1
    SKLEARNSVR = 2
    SKLEARNLINSVC = 3
    SKLEARNLINSVR = 4
    SKLEARNNUSVC = 5
    SKLEARNNUSVR = 6


METAFILE = 'options.json'
MODELPATH = "modelsave/model1"
global mlmodel

global modelformat, trainedmodel, loaded_model


global config
config = {}

   
def process_LinearSVC(modelfile):
    config["name"] = "Testing"
    config["skparams"] = modelfile.get_params()
    if hasattr(modelfile, 'coef_'):
        config["coef_"] = modelfile.coef_.tolist()
    if hasattr(modelfile, 'intercept_'):
        config["intercept_"] = modelfile.intercept_.tolist()
    if hasattr(modelfile, 'classes_'):
        config["classes_"] = modelfile.classes_.tolist()
    if hasattr(modelfile, 'n_features_in_'):
        config["n_features_in_"] = modelfile.n_features_in_ 
    if hasattr(modelfile, 'feature_names_in_'):
        config["feature_names_in_"] = modelfile.feature_names_in_.tolist()
    if hasattr(modelfile, 'n_iter_'):
        config["n_iter_"] = modelfile.n_iter_.item()

    #outJSON = open(FILENAME, WRITEMODE)
    #outJSON.write(json.dumps(config, indent=2))
    #outJSON.close()
    return config

def eval_regression(loaded, dataset):
    X = np.array(np.array(dataset)[:, 1:])
    y = np.array(np.array(dataset)[:, 0])
    predictions = loaded.predict(X)

    
def eval_classification(loaded, dataset):
    X = np.array(np.array(dataset)[:, 1:])
    y = np.array(np.array(dataset)[:, 0])
    predictions = loaded.predict(X)
    conf_matrix = np.array(sklearn.metrics.confusion_matrix(y, predictions))
    loggingdata = dict()

    # Evaluate each class seperately
    for i in range(len(conf_matrix)):
        alle = np.sum(conf_matrix)
        posclass = np.sum(conf_matrix[:, i])
        posreal = np.sum(conf_matrix[i, :])
        negclass = alle - posclass
        negreal = alle - posreal
        tp = conf_matrix[i, i]
        fp = posclass - tp
        tn = alle + tp - posclass - posreal
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


        loggingdata['tp_class' + str(i)] = tp
        loggingdata['fp_class' + str(i)] = fp
        loggingdata['tn_class' + str(i)] = tn
        loggingdata['fn_class' + str(i)] = fn
        loggingdata['tpr_class' + str(i)] = tpr
        loggingdata['tnr_class' + str(i)] = tnr
        loggingdata['ppv_class' + str(i)] = ppv
        loggingdata['npv_class' + str(i)] = npv
        loggingdata['fnr_class' + str(i)] = fnr
        loggingdata['fpr_class' + str(i)] = fpr
        loggingdata['fdr_class' + str(i)] = fdr
        loggingdata['for_class' + str(i)] = fomr
        loggingdata['pos_likelihood_ratio_class' + str(i)] = lrplus
        loggingdata['neg_likelihood_ratio_class' + str(i)] = lrminus
        loggingdata['pt_class' + str(i)] = pt
        loggingdata['ts_class' + str(i)] = ts
        loggingdata['prevalence_class' + str(i)] = posreal/(posreal/negreal)
        loggingdata['accuracy_class' + str(i)] = (tp+tn)/alle
        loggingdata['balanced_accuracy_class' + str(i)] = (tpr + tnr)/2
        loggingdata['f1_score_class' + str(i)] = 2*tp / (2*tp + fp + fn)
        loggingdata['matthews_corrcoef_class' + str(i)] = (tp*tn - fp*fn)/(sqrt(posclass * posreal * negreal * negclass))
        loggingdata['fowlkes_mallows_index_class' + str(i)] = sqrt(ppv*tpr)
        loggingdata['informedness_class' + str(i)] = tpr-tnr-1
        loggingdata['markedness_class' + str(i)] = ppv+npv-1
        loggingdata['diagnostic_odds_ratio_class' + str(i)] = lrplus/lrminus


    # SKlearn scoring for classification:
    loggingdata['balanced_accuracy_score'] = sklearn.metrics.balanced_accuracy_score(y, predictions)
    loggingdata['hinge_loss'] = sklearn.metrics.hinge_loss(y, predictions)
    loggingdata['matthews_corrcoef'] = sklearn.metrics.matthews_corrcoef(y, predictions)
    loggingdata['roc_auc_score'] = sklearn.metrics.roc_auc_score(y, predictions)
    loggingdata['top_two_accuracy_score'] = sklearn.metrics.top_k_accuracy_score(y, predictions, k=2)
    loggingdata['accuracy_score'] = sklearn.metrics.accuracy_score(y, predictions)
    loggingdata['hamming_loss'] = sklearn.metrics.hamming_loss(y, predictions)
    loggingdata['jaccard_score'] = sklearn.metrics.jaccard_score(y, predictions)
    loggingdata['log_loss'] = sklearn.metrics.log_loss(y, predictions)
    loggingdata['zero_one_loss'] = sklearn.metrics.zero_one_loss(y, predictions)
    (precision, recall, f1_score, support) = sklearn.metrics.precision_recall_fscore_support(y, predictions, average='binary')
    loggingdata['precision'] = precision
    loggingdata['recall'] = recall
    loggingdata['f1_score'] = f1_score
    loggingdata['support'] = support
    

    print("Loggings: ", str(loggingdata))

def exec_eval_task(loaded, dataset):

    print("Executing Basic Accuracy Evaluation Task...")
    # prepare training data
    # xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15)
    if modelformat == Modelformat.SKLEARNLINSVC or modelformat == Modelformat.SKLEARNSVC or modelformat == Modelformat.SKLEARNNUSVC:
        return eval_classification(loaded, dataset)

    #elif modelformat == Modelformat.SKLEARNLINSVR or modelformat == Modelformat.SKLEARNSVR or modelformat == Modelformat.SKLEARNNUSVR:
    #    return eval_regression(loaded, dataset)


# Reads the input files

def input_processing(modelpath):
    print("Processing Input Files")
    meta = meta_processing()
#    config = model_processing(meta["modelpath"])
    loaded = model_processing(meta)
    dataset = dataset_processing(meta["datapath"])
    return loaded, dataset

# Reads the model file
def model_processing(modelfile):

    #trainedmodel = joblib.load(modelfile)
    #print("Processing Model Input: " + str(type(trainedmodel)))
    #modeltype = type(trainedmodel)
    loader_module = modelfile['loader_module']

    if(loader_module == "mlflow.sklearn"):
        with open(MODELPATH + "/MLmodel", 'r') as f:
            mlmodel = yaml.load(f, Loader=yaml.FullLoader) # DOPPELT

        modelname = mlmodel['flavors']['sklearn']['pickled_model']
        loaded_model = pickle.load(open(MODELPATH + "/" + modelname, 'rb'))
        modeltype = type(loaded_model)
        global modelformat

        # sklearn SVC Modell
        if modeltype == sklearn.svm._classes.SVC: 
            print('sklearn SVC model detected!')
            modelformat = Modelformat.SKLEARNSVC
            return loaded_model
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
    
    else:
        print("UNSUPPORTED MODEL")
    

    
# Reads the dataset file

def dataset_processing(datafile):
    print("Processing Dataset Input: " + datafile)
    with open(datafile, newline='') as f:
        reader = csv.reader(f)
        dataset = list(reader)
    dataset = [[int(j) for j in i] for i in dataset]
    return dataset

# Reads the meta file

def meta_processing():
    f = open(METAFILE) 
    meta = json.load(f)
    f.close()
    print("Processing Metadata Input: ") # + str(meta))

    with open(MODELPATH + "/MLmodel", 'r') as f:
        mlmodel = yaml.load(f, Loader=yaml.FullLoader)

    modelloader = mlmodel['flavors']['python_function']['loader_module']
    modellpfad = mlmodel['flavors']['python_function']['model_path']

    meta['loader_module'] = modelloader
    meta['model_path'] = modellpfad
    return meta


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [MODELPATH]",
        description="Execute Accuracy Evaluation Task for the [MODELFILE]"
    )

    parser.add_argument(
        "-v", "--version", action="version",
        version=f"{parser.prog} version 0.1.0"
    )

    parser.add_argument('files', nargs='*')
    return parser

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    if not len(args.files)==1:
        print("ERROR: Files expected (1) but given (" + str(len(args.files)) + ")")
        exit(code=1)
    
    modelpath = args.files[0] # unused
    (loaded, dataset) = input_processing(MODELPATH)
#    loaded = input_processing(MODELPATH)[0]
#    dataset = input_processing(MODELPATH)[1]
    exec_eval_task(loaded, dataset)
    print("All Tasks completed")

