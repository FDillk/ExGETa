from io import BytesIO
from pymongo import MongoClient
from bson.objectid import ObjectId
from pprint import pprint
import gridfs
import json

Username = ""
Password = ""

client = MongoClient("mongodb+srv://" + Username + ":" + Password + "@clusterls8.bhtnq.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")

db=client.ma_v1
fs = gridfs.GridFS( db )
files = db.fs.files

def getTaskByID(id):
    return fs.find_one({"_id":  id})

def getTaskByName(name):
    return fs.find_one({"name":  name})

def getApplicableTasks(task, method, type, meta):
    
    metaconditions = [{"metadata.task": task}, {"metadata.method": method},  {"metadata.module": type},
        {"metadata.loader_module": meta["modelmeta"]["loader_module"]},
        {"metadata.classes": meta["datasetmeta"]["n_classes"]},
        {"metadata.os": meta["os"]},
        {"metadata.data_format": meta["datasetmeta"]["data_format"]},
        {"metadata.hardware": meta["hardware"]}]

    tasks = files.find({"$and": metaconditions}, {"_id": 1})
    return tasks

def getTaskFileByID(id):
    task = BytesIO(getTaskByID(id).read())
    return task
