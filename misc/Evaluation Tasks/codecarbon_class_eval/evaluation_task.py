import json
import sklearn
import joblib
import csv
import pickle
from enum import Enum
import mlflow
from codecarbon import EmissionsTracker
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import pandas
import yaml

class Modelformat(Enum):
    SKLEARNSVC = 1
    SKLEARNSVR = 2
    SKLEARNLINSVC = 3
    SKLEARNLINSVR = 4
    SKLEARNNUSVC = 5
    SKLEARNNUSVR = 6
    SKLEARNDTC = 7

CARBONFILE = 'emissions.csv'
METAFILE = 'inputs/options.json'
global mlmodel

global modelformat, trainedmodel, loaded_model

global config
config = {}

#mlflow.set_experiment("my-experiment")

def exec_eval_task(loaded, dataset, metafile):

    print("Executing Codecarbon Evaluation Task...")
    loggingdata = dict()
    
    if(metafile['datasetmeta']['labelloc'] == "f"): # Labels are in first columns
        Y = dataset.iloc[: , :metafile['datasetmeta']["n_labels"]].values
        X = dataset.iloc[: , metafile['datasetmeta']["n_labels"]:].values
    else:                                           # Labels are in last columns
        Y = dataset.iloc[: , -metafile['datasetmeta']["n_labels"]:].values
        X = dataset.iloc[: , :-metafile['datasetmeta']["n_labels"]].values

    if(metafile['datasetmeta']['scaling'] == 'scale'):
        label_encoder = LabelEncoder()
        Y = label_encoder.fit_transform(Y)
        X = StandardScaler().fit_transform(X) 
    
    tracker = EmissionsTracker(output_dir="./")
    tracker.start()
    predictions = loaded.fit(X, Y)
    emissions = tracker.stop()
    
    csvfile = open(CARBONFILE, newline='')
    reader = csv.DictReader(csvfile)
    duration = 0
    emissions_rate = 0
    cpu_power = 0
    gpu_power = 0
    ram_power = 0
    cpu_energy = 0
    gpu_energy = 0
    ram_energy = 0
    energy_consumed = 0
    cpu_count = 0
    for row in reader:
        duration = row['duration']
        emissions = row['emissions']
        emissions_rate = row['emissions_rate']
        cpu_power = row['cpu_power']
        gpu_power = row['gpu_power']
        ram_power = row['ram_power']
        cpu_energy = row['cpu_energy']
        gpu_energy = row['gpu_energy']
        ram_energy = row['ram_energy']
        energy_consumed = row['energy_consumed']
        cpu_count = row['cpu_count']

    loggingdata['duration'] = float(duration)
    loggingdata['emissions'] = float(emissions)
    loggingdata['emissions_rate'] = float(emissions_rate)
    loggingdata['cpu_power'] = float(cpu_power)
    loggingdata['gpu_power'] = float(gpu_power)
    loggingdata['ram_power'] = float(ram_power)
    loggingdata['cpu_energy'] = float(cpu_energy)
    loggingdata['gpu_energy'] = float(gpu_energy)
    loggingdata['ram_energy'] = float(ram_energy)
    loggingdata['energy_consumed'] = float(energy_consumed)
    loggingdata['cpu_count'] = float(cpu_count)

    return loggingdata


# Reads the input files

def input_processing():
    print("Processing Input Files")
    meta = meta_processing()
#    config = model_processing(meta["modelpath"])
    loaded = model_processing(meta)
    dataset = dataset_processing(meta)
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
        print(modeltype)

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
        loaded_model = pickle.load(open(meta['modelmeta']['model_path'], 'rb'))
        modeltype = type(loaded_model)
        print(modeltype)
        
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

def dataset_processing(meta):
    print("Processing Dataset Input...")    
    if(meta['datasetmeta']["data_format"] == "CSV"):
        dataset = pandas.read_csv(meta['datasetmeta']["data_path"], sep=meta['datasetmeta']["data_separator"])
    else:
        print("ERROR: Unsupported Dataset Format")
        dataset = 0

    return dataset

# Reads the meta file

def meta_processing():
    f = open(METAFILE) 
    meta = json.load(f)
    f.close()
    print("Processing Metadata Input...")
    
    return meta

def metricsaver(tolog, meta):
    print("Saving metrics to MLflow...")
    mlflow.set_tracking_uri(meta["mlflow_uri"])
    with mlflow.start_run(run_name='Codecarbon Efficiency Evaluation'):
        mlflow.log_metrics(tolog)
        mlflow.log_artifact("emissions.csv")


if __name__ == "__main__":
    
    (loaded, dataset, meta) = input_processing()
    
    tolog = exec_eval_task(loaded, dataset, meta)
    print("All Tasks completed")
    metricsaver(tolog, meta)

