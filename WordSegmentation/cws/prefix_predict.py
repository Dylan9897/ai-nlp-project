"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/12/13 15:05
@Email : handong_xu@163.com
"""
from module.log import logger
from module.config import Config
from models.dag import CWS,DAG
from module.dataloader import loading_vocab

config = Config()

def vio_cut(dic,seqList):
    fun = DAG(dic)
    cws = CWS(dic)
    logger.info("正在生成句子的dag图...")
    seq_dag_res = fun.dag(seqList)
    logger.info("句子的dag图生成完毕...")
    result = cws.violent_cut(seq_dag_res,seqList)
    return result

def nbr_matrix_cut(dic,seqList):
    cws = CWS(dic)
    logger.info("正在生成句子的dag图的邻接矩阵...")
    result = cws.leighbor_matrix_cut(dic,seqList)
    return result

if __name__ == '__main__':
    seqList = ['共同创造美好新世纪',
               '女士们，先生们，同志们，朋友们：']
    prefix_vocab = loading_vocab(config.vocab_file)
    res = nbr_matrix_cut(prefix_vocab,seqList)
    print(res)
    print('='*50)
    res = vio_cut(prefix_vocab,seqList)
