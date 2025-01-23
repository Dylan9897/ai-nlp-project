import os
import shutil

class DAGMProcess():
    def __init__(self):
        self.root = "data/dagm_2017"
        self.train_path = "data/dagm_2017/train"
        self.test_path = "data/dagm_2017/test"
        if not os.path.exists(self.train_path):
            os.mkdir(self.train_path)
        if not os.path.exists(self.test_path):
            os.mkdir(self.test_path)

    # 读取label.txt文件
    def _read_label_path(self,path):
        good = []
        broken = []
        with open(path,"r",encoding="utf-8") as fl:
            for i,line in enumerate(fl.readlines()):
                if i == 0:
                    continue
                line = line.strip("\ufeff").strip("\n").split("\t")
                if line[1] == "0" and line[-2] == "0":
                    good.append(line[2])
                else:
                    broken.append(line[2])
        return good,broken

    # 生成测试集和训练集
    def _generate(self):
        file_list_l1 = os.listdir(self.root)
        for i,root_l1 in enumerate(file_list_l1):
            if not str(root_l1).startswith("Class"):
                continue
            
            file_path_l1 = os.path.join(os.path.join(self.root,root_l1),root_l1)
            print(file_path_l1)
            train_file_path = os.path.join(file_path_l1,"Train")
            test_file_path = os.path.join(file_path_l1,"Test")
            ## 处理训练集
            train_label_file_path = os.path.join(train_file_path,"Label/Labels.txt")
            train_good,train_broken = self._read_label_path(train_label_file_path)
            for unit in train_good:
                aim_file_path = os.path.join(train_file_path,unit)
                shutil.copy(aim_file_path,self.train_path)

            ## 处理测试集
            test_label_file_path = os.path.join(test_file_path,"Label/Labels.txt")
            test_good,test_broken = self._read_label_path(test_label_file_path)
            for unit in test_good:
                aim_file_path = os.path.join(test_file_path,unit)
                shutil.copy(aim_file_path,os.path.join(self.test_path,"good"))
            for unit in test_broken:
                aim_file_path = os.path.join(test_file_path,unit)
                shutil.copy(aim_file_path,os.path.join(self.test_path,"broken"))

class MVTECProcess():
    def __init__(self):
        self.root = "data/MVTEC"
        self.train_path = "data/MVTEC/train"
        self.test_path = "data/MVTEC/test"
        if not os.path.exists(self.train_path):
            os.mkdir(self.train_path)
        if not os.path.exists(self.test_path):
            os.mkdir(self.test_path)

    # 生成训练集和测试集
    def _generate(self):
        i,k = 0,0
        file_list_l1 = os.listdir(self.root)
        for file_path in file_list_l1:
            if file_path.endswith("txt") or file_path in ["train","test"]:
                continue
            file_root_l1 = os.path.join(self.root,file_path)
            print(file_root_l1)
            # 处理训练集
            train_path = os.path.join(file_root_l1,"train/good")
            train_file_list = os.listdir(train_path)
            for unit in train_file_list:
                shutil.copy(os.path.join(train_path,unit),self.train_path+"/00{}.png".format(i))
                i+=1

            # 处理测试集
            test_path = os.path.join(file_root_l1,"test")
            for test_root_1 in os.listdir(test_path):
                test_root_path = os.path.join(test_path,test_root_1)
                if test_root_1 == "good":
                    test_file_list = os.listdir(test_root_path)
                    for unit in test_file_list:
                        if unit.endswith("txt"):
                            continue
                        shutil.copy(os.path.join(test_root_path,unit),os.path.join(self.test_path,"good"+"/00{}.png".format(k)))
                        k+=1
                else:
                    test_file_list = os.listdir(test_root_path)
                    for unit in test_file_list:
                        if unit.endswith("txt"):
                            continue
                        shutil.copy(os.path.join(test_root_path,unit),os.path.join(self.test_path,"broken"+"/00{}.png".format(k)))
                        k+=1
            
if __name__=="__main__":
    fun = MVTECProcess()
    fun._generate()



