import json
from math import sqrt
import sklearn
import os
import joblib
import pickle
import yaml
from enum import Enum
import mlflow
import joblib
import pandas
from classifier_mapper import ClassifierMapper
import subprocess

class Modelformat(Enum):
    SKLEARNSVC = 1
    SKLEARNSVR = 2
    SKLEARNLINSVC = 3
    SKLEARNLINSVR = 4
    SKLEARNNUSVC = 5
    SKLEARNNUSVR = 6

CARBONFILE = 'emissions.csv'
METAFILE = 'inputs/options.json'
global mlmodel

global modelformat, trainedmodel, loaded_model

global config
config = {}


def exec_eval_task(loaded, dataset, meta):

    print("Executing SAVer Robustness Evaluation Task...")
    loggingdata = dict()
    ClassifierMapper.createSvm(ClassifierMapper, loaded, "./converted.pkl")
    cwd = os.path.abspath(os.getcwd()).replace("\\", "/")
    newmodelpath = os.path.abspath("./converted.pkl")
    fulldatapath = os.path.abspath(meta['datasetmeta']['data_path'])

    # 60000 784
    with open(meta['datasetmeta']['data_path'], 'r') as file:
        # read a list of lines into data
        data = file.readlines()

    # Change first line for Saver Dataformat to number of examples and features
    data[0] = '# 350 34\n'

    # and write everything back
    with open(meta['datasetmeta']['data_path'], 'w') as file:
        file.writelines( data )

    p = subprocess.Popen(['C:/cygwin64/bin/bash.exe', '-c', '. /etc/profile; cd ' + cwd + '; saver/bin/saver ./converted.pkl ' + meta['datasetmeta']['data_path'] + ' interval l_inf 0.01'], 
          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    result = p.communicate()[0]
    #print(result)
    lastrow = (str(result, 'utf-8').splitlines())[len(str(result, 'utf-8').splitlines()) - 1]

    siz = float(lastrow.split()[1])
    loggingdata['epsilon'] = float(lastrow.split()[2])
    loggingdata['avgtime'] = float(lastrow.split()[3])
    ncorrect = float(lastrow.split()[4])
    nrobust = float(lastrow.split()[5])
    ncondrobust = float(lastrow.split()[6])

    loggingdata['ratiocorrect'] = ncorrect/siz
    loggingdata['ratiorobust'] = nrobust/siz
    loggingdata['ratiocondrobust'] = ncondrobust/siz
    loggingdata['size'] = siz
    loggingdata['ncorrect'] = ncorrect
    loggingdata['nrobust'] = nrobust
    loggingdata['nconditionalrobust'] = ncondrobust

    return loggingdata


# Reads the input files

def input_processing():
    print("Processing Input Files")
    meta = meta_processing()
#    config = model_processing(meta["modelpath"])
    loaded = model_processing(meta)
    dataset = dataset_processing(meta)
#    dataset = meta["datapath"]
    return loaded, dataset, meta

# Reads the model file
def model_processing(meta):

    #trainedmodel = joblib.load(modelfile)
    #print("Processing Model Input: " + str(type(trainedmodel)))
    #modeltype = type(trainedmodel)
    loader_module = meta['modelmeta']['loader_module']

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
            # infos = process_LinearSVC(loaded_model)
        else:
            print('Unsupported sklearn model! modeltype: ' + modeltype)

        return loaded_model
    elif(loader_module == "sklearn.joblib"):
        loaded_model = joblib.load(open(meta['modelmeta']['model_path'], 'rb'))
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
            # infos = process_LinearSVC(loaded_model)
        
        return loaded_model
    elif(loader_module == "sklearn.pickle"):
        loaded_model = pickle.load(open(meta['modelmeta']['model_path'], 'rb'))
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
            # infos = process_LinearSVC(loaded_model)
        
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
    f = open(METAFILE) 
    meta = json.load(f)
    f.close()
    print("Processing Metadata Input: ") # + str(meta))

    return meta

def metricsaver(tolog, meta):
    print("Saving metrics to MLflow...")
    mlflow.set_tracking_uri(meta["mlflow_uri"])
    with mlflow.start_run(run_name='SAVer Robustness Evaluation'):
        mlflow.log_metrics(tolog)

if __name__ == "__main__":
    
    (loaded, dataset, meta) = input_processing()
    
    tolog = exec_eval_task(loaded, dataset, meta)
    print("All Tasks completed")
    metricsaver(tolog, meta)

