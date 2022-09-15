
import mlflow
import json

WRITEMODE = 'w'
FILENAME = 'OUT.json'

global config
config = {}

def process_ALL(modelfile):
    config["name"] = "Testing"
    config["mlfparams"] = modelfile.get_model_info()    

    #outJSON = open(FILENAME, WRITEMODE)
    #outJSON.write(json.dumps(config, indent=2))
    #outJSON.close()
    return config
