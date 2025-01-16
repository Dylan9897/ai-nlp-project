"""
预处理数据集
"""
import os
import json
import pandas as pd
import argparse

import shutil
from sklearn.model_selection import train_test_split

def operate_thucnews(src,dst):
    if not os.path.exists(dst):
        os.makedirs(dst)

    # 移动 标签文件
    shutil.copy(os.path.join(src,"class.txt"),dst)

    # 写入训练集、验证集、测试集
    for tag in ["train","dev","test"]:
        file_path = os.path.join(src,"{}.txt".format(tag))
        with open(file_path,"r",encoding="utf-8") as fl:
            with open(os.path.join(dst,"{}.json".format(tag)),"w",encoding="utf-8") as ft:

                for i,line in enumerate(fl.readlines()):
                    line = line.strip("\n").split("	")
                    data = {
                        "metadata":i+1,
                        "content":line[0],
                        "label":line[1]
                    }
                    json_data = json.dumps(data,ensure_ascii=False)
                    ft.write(json_data+'\n')

def operate_longnews(src,dst):
    df = pd.read_csv(os.path.join(src,"labeled_data.csv"))
    train,dev = train_test_split(df,test_size=0.143,random_state=2025)
    unique_values = train['class_label'].unique()
    label_mapping = {k:str(v) for v,k in enumerate(unique_values)}
    with open(os.path.join(dst,"class.txt"),"w",encoding="utf-8") as ft:
        ft.write("\n".join(unique_values))

    # print(dev["class_label"].value_counts())
    with open(os.path.join(dst,"{}.json".format("train")),"w",encoding="utf-8") as ft:
        for i in train.index:
            line = train.loc[i]
            data = {
                "metadata":str(line["id"]),
                "content":line["content"],
                "label":label_mapping[line["class_label"]]
            }
            json_data = json.dumps(data,ensure_ascii=False)
            ft.write(json_data+"\n")

    with open(os.path.join(dst, "{}.json".format("dev")), "w", encoding="utf-8") as ft:
        for i in dev.index:
            line = dev.loc[i]
            data = {
                "metadata": str(line["id"]),
                "content": line["content"],
                "label": label_mapping[line["class_label"]]
            }
            json_data = json.dumps(data, ensure_ascii=False)
            ft.write(json_data + "\n")



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, default='thucnews', help='dataset name')
    parser.add_argument("--output", type=str, default='./data/datasets', help='data path')
    args = parser.parse_args()
    
    if args.dataset == 'thucnews':
        operate_thucnews("data/THUCNews/data",os.path.join(args.output, 'thucnews'))

    elif args.dataset == 'longnews':
        operate_longnews("data/长文本分类",os.path.join(args.output,"longnews"))

