"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/12/13 15:35
@Email : handong_xu@163.com
"""
import pickle
from models.hmm import HMM,viterbi



if __name__ == '__main__':
    model = HMM()
    model.init_matrix,model.transfer_matrix,model.emit_matrix = pickle.load(open('ckpt/hmm_cws.pkl','rb'))
    seqList = ['共同创造美好新世纪',
               '女士们，先生们，同志们，朋友们：']
    for line in seqList:
        result = viterbi(line,model)
        print(result)