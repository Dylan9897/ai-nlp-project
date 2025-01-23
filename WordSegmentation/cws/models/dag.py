"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/12/13 15:13
@Email : handong_xu@163.com
"""
import copy
from module.log import logger

class DAG():
    """
    根据前缀字典生成dag图
    """
    def __init__(self,dic):
        self.dic = dic

    def dag_matrix(self,seq):
        """
        生成当前句子的转移矩阵
        :param seq:
        :return:
        """
        m = len(seq)
        matrix = [[0] * m for _ in range(m)]
        for i in range(m):
            matrix[i][i] = 1
            for j in range(i, m):
                if seq[i:j] in self.dic:
                    matrix[i][j - 1] = 1
        return matrix

    def dag(self,seqList):
        """
        生成当前句子的dag图
        :param seq:
        :return:
        """
        final = {}
        for i,seq in enumerate(seqList):
            res = {}
            matrix = self.dag_matrix(seq)
            for i,unit in enumerate(matrix):
                temp = []
                for e,index in enumerate(unit):
                    if index!=0:
                        temp.append(e)
                res[i] = temp
            final['seq{}'.format(i)] = res
        return final

class CWS(DAG):
    def __init__(self,dic):
        super(CWS, self).__init__(dic)
        self.dic = dic

    def merge_all_index(self,dic):
        """
        :param dic: {0: [0], 1: [1, 2, 4], 2: [2], 3: [3, 4], 4: [4], 5: [5]}
        :return: [[0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 5], [0, 2, 3, 4, 5], [0, 2, 4, 5], [0, 4, 5]]
        """
        def helper(e):
            if e >= len(dic):
                final.append(copy.deepcopy(temp))
            else:
                for unit in dic[e]:
                    temp.append(unit)
                    helper(unit+1)
                    temp.pop()
        temp = []
        final = []
        helper(0)
        return final

    def merge_seq_cws(self,seq,temp):
        """
        :param seq: '去北京大学玩'
        :param temp: [0, 1, 2, 3, 4, 5]
        :return:
        """
        res = []
        i,j = 0,0
        word = ''
        while i<len(seq):
            word += seq[i]
            if i == temp[j]:
                res.append(word)
                word = ''
                j+=1
            i+=1
        return res

    def get_all_cws(self,seq,temp):
        """
        生成所有句子的候选
        :param seq:
        :param temp:
        :return:
        """
        final = []
        for unit in temp:
            res = self.merge_seq_cws(seq,unit)
            final.append(res)
        return final

    def calc(self,cws_list,word_dic):
        """
        计算最优的组合
        :param cws_list:
        :param word_dic:
        :return:
        """
        key = 0
        flag = 0
        cur = 0
        for i,unit in enumerate(cws_list):
            for word in unit:
                k = word_dic.get(word,0)
                cur+=int(k)
            if cur > key:
                flag = i
                key = cur
        return cws_list[flag]



    def violent_cut(self,dag_res,seqList):
        """
        暴力穷举所有可能的组合
        """
        final = {}
        seq_res = list(dag_res.items())
        for i,seq in enumerate(seqList):
            res=seq_res[i][1]
            logger.info("Start merge all index...")
            res = self.merge_all_index(res)
            logger.info("Finish merge all index...")
            logger.info("正在生成当前句子所有可能的分词结果...")
            cws_all = self.get_all_cws(seq,res)
            logger.info("当前句子所有可能的分词结果生成完毕")
            logger.info("计算最优分词结果...")
            best = self.calc(cws_all,self.dic)
            logger.info("最优分词结果计算完毕")
            final['seq{}'.format(i)] = best
        return final


    def leighbor_matrix_cut_helper(self,dic,seq):
        final = []
        leighbor_matrix = self.dag_matrix(seq)
        i=0
        while i<len(leighbor_matrix):
            j=i
            key = 0
            cur_word = ""
            idx = 0
            while j <len(leighbor_matrix):
                if leighbor_matrix[i][j] == 1:
                    if i == j:
                        word = seq[i]
                    else:
                        word = seq[i:j+1]
                    cur = int(dic.get(word,0))
                    if cur >=key :
                        cur_word = word
                        key=cur
                        idx = j
                j+=1
            final.append(cur_word)
            i=idx
            i+=1
        return final

    def leighbor_matrix_cut(self,dic,seqList):
        """
        使用动态规划算法，在dag图邻接矩阵，求取最优值
        """
        final = []
        for seq in seqList:
            res = ' '.join(self.leighbor_matrix_cut_helper(dic,seq))
            final.append(res)
        return final

