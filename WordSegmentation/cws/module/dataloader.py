"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/12/13 14:47
@Email : handong_xu@163.com
"""



def merge_prefix_vocab(src:str,dst:str) -> None:
    """
    生成前缀字典
    :param src:  文件路径
    :param dst:  文件路径
    :return:
    """
    dic = {}
    with open(src,'r',encoding='utf-8') as fl:
        fl = fl.read().split('\n')
        for line in fl:
            line = line.split('  ')
            for unit in line:
                if unit == '':
                    continue
                dic[unit] = dic.get(unit,0)+1
    with open(dst,'w',encoding='utf-8') as ft:
        for unit in dic.items():
            ft.write(str(unit[0])+'\t'+str(unit[1])+'\n')


def sorted_vocab(dic):
    """
    对字典进行排序，将第一个字相同的排在一起
    :param dic:
    :return:
    """
    res = {}
    word_list = sorted(dic.items(),key=lambda x:(x[0][0],-len(x[0])))
    for unit in word_list:
        # print(unit)
        res[unit[0]] = unit[1]
    return res


def loading_vocab(path):
    """
    读取词表
    :param path:
    :return:
    """
    dic = {}
    with open(path, 'r', encoding='utf-8') as fl:
        fl = fl.read().split('\n')
        for line in fl:
            if line == '':
                continue
            line = line.split('\t')
            dic[line[0]] = line[1]
    dic = sorted_vocab(dic)
    return dic


def read_file(file):
    with open(file,'r',encoding='utf-8') as fl:
        fl = fl.read().split('\n')
    return fl

def trans_helper(seq):
    res = 'B'
    for i in range(1,len(seq)-1):
        res+='M'
    res+='E'
    return res

def trans_hmm(fileList,dst):
    """
    生成BIEM标注模式的数据集
    :param fileList:
    :param dst:
    :return:
    """
    with open(dst,'w',encoding='utf-8') as ft:
        for seq in fileList:
            tag = ''
            seq = seq.split('  ')
            for unit in seq:
                if len(unit) == 1:
                    tag+='S'+' '
                elif len(unit) == 2:
                    tag+='BE'+' '
                else:
                    curtag=trans_helper(unit)
                    tag+=curtag+' '
            ft.write(tag+'\n')






