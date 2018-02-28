from os.path import *
from os import listdir

def fileparts(path): # path = file path
    [p,f] = split(path)
    [n,e] = splitext(f)
    return [p,n,e]

def listfiles(path,token): # path = folder path
    l = []
    for f in listdir(path):
        fullPath = join(path,f)
        if isfile(fullPath) and token in f:
            l.append(fullPath)
    return l

def pathjoin(p,ne): # '/path/to/folder', 'name.extension'
    return join(p,ne)