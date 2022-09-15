import argparse
import zipfile
import joblib
import sklearn
import json
import time
import skprocessing as skp
import dbconnector as dbc
import mlflow
import mlflow.sklearn
import mlprocessing
import shutil
import os

WRITEMODE = 'w'
FILENAME = 'options.json'

global model, meta, data, constraints
global task, method, mtype
global modelfile, metafile, datafile
model = ""

# Reads the input files

def input_processing(modelf, metaf, dataf):
    global modelfile, metafile, datafile
    modelfile = modelf
    datafile = dataf

    print("Processing Input Files")
    meta = meta_processing(metaf)
    modelinfo = model_processing(modelf)
    dataset_processing(dataf)
    print("Processing Input Files done!")
    print("Creating Config JSON")
    meta["generatedmodelinfo"] = modelinfo

    outJSON = open(FILENAME, WRITEMODE)
    outJSON.write(json.dumps(meta, indent=2))
    outJSON.close()
    print("Creating Config JSON done!")

# Reads the model file 
def model_processing(modelfile):
    global task, method, mtype
    model = joblib.load(modelfile)
    print("Processing Model Input: " + str(type(model)))
    modeltype = type(model)
    time.sleep(1)

    # MLFlow Modell
    if modeltype == mlflow.models.Model: 
        print('MLflow model detected!')
        return mlprocessing.process(model)

    # sklearn SVC Modell
    elif modeltype == sklearn.svm._classes.SVC: 
        print('sklearn SVC model detected!')
        task = "classification"
        method = "SVM"
        mtype = "sklearn.SVC"
        return skp.process_SVC(model)

    # sklearn LinearSVC Modell
    elif modeltype == sklearn.svm._classes.LinearSVC:
        print('sklearn LinearSVC model detected!')
        task = "classification"
        method = "SVM"
        mtype = "sklearn.linearSVC"
        return skp.process_LinearSVC(model)

    # sklearn NuSVC Modell
    elif modeltype == sklearn.svm._classes.NuSVC:
        print('sklearn NuSVC model detected!')
        task = "classification"
        method = "SVM"
        mtype = "sklearn.NuSVC"
        return skp.process_NuSVC(model)

    # sklearn SVR Modell
    elif modeltype == sklearn.svm._classes.SVR:
        print('sklearn SVR model detected!')
        task = "regression"
        method = "SVM"
        mtype = "sklearn.SVR"
        return skp.process_SVR(model)

    # sklearn Linear SVR Modell
    elif modeltype == sklearn.svm._classes.LinearSVR:
        print('sklearn Linear SVR model detected!')
        task = "regression"
        method = "SVM"
        mtype = "sklearn.linearSVR"
        return skp.process_LinearSVR(model)

    # sklearn NuSVR Modell
    elif modeltype == sklearn.svm._classes.NuSVR:
        print('sklearn NuSVR model detected!')
        task = "regression"
        method = "SVM"
        mtype = "sklearn.NuSVR"
        return skp.process_NuSVR(model)

    # Unknown
    else:
        print("No supported model found!")
        print("modeltype:")
        print(modeltype)
        exit(1)

# Reads the dataset file

def dataset_processing(datafile):
    print("Processing Dataset Input: " + datafile)
    # Dataset processing not needed yet

# Reads the meta file

def meta_processing(metafile):
    print("Processing Metadata Input")
    f = open(metafile) 
    meta = json.load(f)
    f.close()
    return meta

def build_tasks(task, method, mtype, metafile):

    ts = dbc.getApplicableTasks(task, method, mtype, metafile)
    tn = 0
    for t in ts:
        tpath = metafile["outpath"] + "/eval_tasks/task" + str(tn)
        zip = dbc.getTaskFileByID(t["_id"])
        with zipfile.ZipFile(zip, 'r') as toextract:
            toextract.extractall(path=(tpath))
            os.mkdir(tpath + "/inputs/")
            shutil.copy(modelfile, tpath + "/inputs/" + os.path.basename(modelfile))
            shutil.copy('options.json', tpath + "/inputs/options.json")
            shutil.copy(datafile, tpath + "/inputs/" + os.path.basename(datafile))
        tn = tn + 1
    print(str(tn) + " tasks have been generated!")
# Processing of Script Call

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [METADATA]",
        description="Generate Evaluation Tasks for a model, providing the required [METADATA]"
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
    
    # Speichern der Argumente (Pfad zum Modell, Pfad zur Metadatei, Pfad zum Datensatz)
    metapath = args.files[0]
    metajson = meta_processing(metapath)
    modelfile = metajson["modelmeta"]["model_path"]
    datafile = metajson["datasetmeta"]["data_path"]
    input_processing(modelfile, metapath, datafile)

    print("Generating tasks....")
    time.sleep(2)
    build_tasks(task, method, mtype, metajson)

    print("Task generation complete!")
    print("Tasks can be found in the specified directory")
