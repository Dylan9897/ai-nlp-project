"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/4/19 10:52
@Email : handong_xu@163.com
"""
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

token = lambda x:list(x)

def read_file(file):
    df = pd.read_csv(file)
    df = shuffle(df)
    print(f"value counts is {df['label'].value_counts()}")
    print(df["label"].describe())
    label_id2cate = dict(enumerate(df.label.unique()))
    label_cate2id = {value:key for key,value in label_id2cate.items()}
    df["tag"] = df["label"].map(label_cate2id)
    return df

    

def get_data(file):
    df = read_file(file)
    X_train,X_test,y_train,y_test = train_test_split(df['contents'],df['tag'],test_size=0.05,stratify=df['tag'])
    return X_train,X_test,y_train,y_test



if __name__=="__main__":
    file = "data/非本人.csv"
    df = get_data(file)
    



