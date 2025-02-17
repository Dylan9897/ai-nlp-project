import pickle

def write_pkl_file(obj,path):
    with open(path,'wb') as ft:
        pickle.dump(obj,ft)

def read_pkl_file(path):
    with open(path,'rb') as fl:
        obj = pickle.load(fl)
    return obj

