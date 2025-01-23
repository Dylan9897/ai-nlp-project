"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/12/13 15:32
@Email : handong_xu@163.com
"""
import os
import pickle
from tqdm import tqdm
import numpy as np


class HMM(object):
    def __init__(self,file_text=None,file_state=None):
        if file_text != None or file_state != None:
            self.all_states = open(file_state,'r',encoding='utf-8').read().split('\n')
            self.all_texts = open(file_text,'r',encoding='utf-8').read().split('\n')

        self.states_to_idnex = {'B':0,'M':1,'S':2,'E':3}
        self.index_to_state = ['B','M','S','E']  # 观测集合
        self.len_states = len(self.states_to_idnex)

        self.init_matrix = np.zeros(self.len_states) # 初始矩阵
        self.transfer_matrix = np.zeros((self.len_states,self.len_states)) # 转移矩阵

        self.emit_matrix = {"B":{"total":0},"M":{"total":0},"S":{"total":0},'E':{'total':0}}  # 发射矩阵

    def cal_init_matrix(self,state):
        """
        计算初始矩阵
        """
        self.init_matrix[self.states_to_idnex[state[0]]]+=1

    def cal_transfer_matrix(self,state):
        sta_join = ''.join(state)
        sta1 = sta_join[:-1]
        sta2 = sta_join[1:]
        for s1,s2 in zip(sta1,sta2):
            self.transfer_matrix[self.states_to_idnex[s1],self.states_to_idnex[s2]] +=1

    # 计算发射矩阵
    def cal_emit_matrix(self, words, states):
        for word, state in zip("".join(words), "".join(states)):  # 先把words 和 states 拼接起来再遍历, 因为中间有空格
            self.emit_matrix[state][word] = self.emit_matrix[state].get(word, 0) + 1
            self.emit_matrix[state]["total"] += 1  # 注意这里多添加了一个  total 键 , 存储当前状态出现的总次数, 为了后面的归一化使用

    # 将矩阵归一化
    def normalize(self):
        self.init_matrix = self.init_matrix / np.sum(self.init_matrix)
        self.transfer_matrix = self.transfer_matrix / np.sum(self.transfer_matrix, axis=1, keepdims=True)
        self.emit_matrix = {
            state: {word: t / word_times["total"] * 1000 for word, t in word_times.items() if word != "total"} for
            state, word_times in self.emit_matrix.items()}


    def train(self):
        if os.path.exists('ckpt/hmm_cws.pkl'):
            self.init_matrix,self.transfer_matrix,self.emit_matrix = pickle.load(open('ckpt/hmm_cws.pkl','rb'))
            return
        for words,states in tqdm(zip(self.all_texts,self.all_states)):
            words = words.split(' ')
            states = states.split(' ')
            self.cal_init_matrix(states[0])
            self.cal_transfer_matrix(states)
            self.cal_emit_matrix(words, states)
        self.normalize()
        pickle.dump([self.init_matrix,self.transfer_matrix,self.emit_matrix],open('ckpt/hmm_cws.pkl','wb'))


def viterbi(text, hmm):
    states = hmm.states_to_idnex
    emit_p = hmm.emit_matrix
    trans_p = hmm.transfer_matrix
    start_p = hmm.init_matrix
    V = [{}]
    path = {}
    for y in states:
        V[0][y] = start_p[hmm.states_to_idnex[y]] * emit_p[y].get(text[0], 0)
        path[y] = [y]
    for t in range(1, len(text)):
        V.append({})
        newpath = {}
        # 检验训练的发射概率矩阵中是否有该字
        neverSeen = text[t] not in emit_p['S'].keys() and \
                    text[t] not in emit_p['M'].keys() and \
                    text[t] not in emit_p['E'].keys() and \
                    text[t] not in emit_p['B'].keys()
        for y in states:
            emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0  # 设置未知字单独成词
            temp = []
            for y0 in states:
                if V[t - 1][y0] > 0:
                    temp.append((V[t - 1][y0] * trans_p[hmm.states_to_idnex[y0],hmm.states_to_idnex[y]] * emitP, y0))
            (prob, state) = max([(V[t - 1][y0] * trans_p[hmm.states_to_idnex[y0],hmm.states_to_idnex[y]] * emitP, y0)  for y0 in states if V[t - 1][y0] > 0])
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath


    (prob, state) = max([(V[len(text) - 1][y], y) for y in states])  # 求最大概念的路径

    result = "" # 拼接结果
    for t,s in zip(text,path[state]):
        result += t
        if s == "S" or s == "E" :  # 如果是 S 或者 E 就在后面添加空格
            result += " "
    return result