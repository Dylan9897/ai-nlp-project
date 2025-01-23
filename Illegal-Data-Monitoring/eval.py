import os
from predict import api
from tqdm import tqdm
from sklearn.metrics import classification_report,accuracy_score


from loguru import logger
 
logger.remove(handler_id=None)


fun = api()

def eval1():
    y_true = []
    y_pred = []
    file_list = os.listdir("data/ood/test")
    for unit in tqdm(file_list):
        file_path = os.path.join("data/ood/test",unit)
        pred = fun.predict_raw(file_path)
        if pred==True:
            y_pred.append(1)
        else:
            y_pred.append(0)
        y_true.append(1)
    print(y_pred)
    acc = accuracy_score(y_true,y_pred)
    clr = classification_report(y_true,y_pred)
    print(acc)
    print(clr)

def eval2():
    y_true = []
    y_pred = []
    file_list = os.listdir("data/MVTEC/test/broken")
    for unit in tqdm(file_list):
        file_path = os.path.join("data/MVTEC/test/broken",unit)
        pred = fun.predict_raw(file_path)
        if pred==True:
            y_pred.append(1)
        else:
            y_pred.append(0)
        y_true.append(1)

    file_list = os.listdir("data/MVTEC/test/good")
    for unit in tqdm(file_list):
        file_path = os.path.join("data/MVTEC/test/good",unit)
        pred = fun.predict_raw(file_path)
        if pred==True:
            y_pred.append(1)
        else:
            y_pred.append(0)
        y_true.append(0)
    print(y_pred)
    s = input('################')
    print(y_true)
    acc = accuracy_score(y_true,y_pred)
    clr = classification_report(y_true,y_pred)
    print(acc)
    print(clr)

if __name__ == "__main__":
    # fun = api()
    # example = "data/ood/test/a0a34a4f1302e484066e4934c3008803.jpg"
    # print(fun.predict_raw(example))
    eval2()


