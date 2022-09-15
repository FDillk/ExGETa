from pymongo import MongoClient
from bson.objectid import ObjectId
from pprint import pprint
import gridfs

client = MongoClient("mongodb+srv://fabiandillkoetter:AutomatedGenerationMA@clusterls8.bhtnq.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")

db=client.ma_v1
collection=db.evaluation_codes
fs = gridfs.GridFS( db )


meta = {
    "version": "1.0",
    "task": ["classification", "regression"],
    "classes": ["2", "multi"],
    "os": ["windows", "unix", "macos"],
    "hardware": ["cpu"],
    "data_format": ["CSV"],
    "method": ["SVM", "DecisionTree"],
    "module": ["sklearn.SVC", "sklearn.LinearSVC", "sklearn.NuSVC", "sklearn.DecisionTree"],
    "loader_module": ["sklearn.pickle", "sklearn.joblib"]
},

fileID = fs.put( open( r'evaluation_task.zip', 'rb'), metadata = meta, name="codecarbon_eval"  )

def getCodeByID(id):
    return collection.find_one({"_id":  ObjectId(id)})

def getCodeByName(name):
    return collection.find_one({"name":  name})
