import os

def getRootPath():

    rootPath=os.path.dirname(os.path.abspath(__file__))

    rootPath=os.path.dirname(rootPath)
    # rootPath=rootPath.split("/DAM/config")[0]

    return rootPath