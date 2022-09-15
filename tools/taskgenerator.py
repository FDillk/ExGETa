import dbconnector

def generateTasksByIDs(ids):
    for id in ids:
        code = dbconnector.getCodeByID(id)
