"""
-*- coding: utf-8 -*-

@Author : dongdong
@Time : 2022/4/20 18:51
@Email : handong_xu@163.com
"""

def read_file():
    vocab = set([line.strip() for line in open('data/vocab.txt')])
    return vocab
vocab = read_file()

## 生成编辑距离为1的候选单词集合
def generate_edit_one(wrong_word,flag=True):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(wrong_word[:i],wrong_word[i:]) for i in range(len(wrong_word)+1)]
    inserts = [left+letter+right for left,right in splits for letter in letters]
    deletes = [left+right[1:] for left,right in splits]
    replaces = [left+letter+right[1:] for left,right in splits for letter in letters]
    candidates = set(inserts+deletes+replaces)
    if flag:
        return candidates
    return [candi for candi in candidates if candi in vocab]

## 生成编辑距离为2的候选单词集合
def generate_edit_two(wrong_word):
    candi_one = generate_edit_one(wrong_word)
    candi_list = []
    for candi in candi_one:
        candi_list.extend(generate_edit_one(candi,flag=False))
    candi_two = set(candi_list)
    return [candi for candi in candi_two if candi in vocab]

## 加载拼写错误的文件，统计正确单词被拼写成不同错误单词的次数
misspell_prob = {}
for line in open('data/spell-errors.txt'):
    items = line.split(':')
    correct = items[0].strip()
    misspells = [item.strip() for item in items[1].split(',')]
    misspell_prob[correct] = {}
    for misspell in misspells:
        misspell_prob[correct][misspell] = 1/len(misspells)


## 加载语料库，统计正确单词出现在一句话中的次数，使用Bigram语言模型
with open('data/news.en','r',encoding='utf-8') as fl:
    corpus = fl.read().split('\n')
corpus = [line.split(' ') for line in corpus]
# 构建语言模型：bigram
term_count = {}
bigram_term_count = {}
for doc in corpus:
    doc = ['<s>']+doc
    for i in range(len(doc)-1):
        term = doc[i]
        bigram_term = doc[i:i+2]
        bigram_term = ' '.join(bigram_term)
        # print(bigram_term)
        if term in term_count:
            term_count[term] += 1
        else:
            term_count[term] = 1
        if bigram_term in bigram_term_count:
            bigram_term_count[bigram_term] += 1
        else:
            bigram_term_count[bigram_term] = 1

## 加载测试数据，找出拼写错误的单词，生成候选词并计算每个候选词的概率，找出概率最大的候选词作为正确的单词
import numpy as np
V = len(term_count)

with open('data/testdata.txt','r',encoding='utf-8') as fl:
    for line in fl:
        items = line.split('\t')
        word_list = items[2].split()
        print(word_list)
        for index,word in enumerate(word_list):
            word = word.strip(',.')
            if word not in vocab:
                candidates = generate_edit_one(word,flag=False)
                if len(candidates) == 0:
                    candidates = generate_edit_two(word)
                probs = []
                prob_dict = {}
                # 对于每一个candidate，计算它的prob，
                # prob = p(correct)*p(mistake|correct)
                # =logP(correct)+logP(mistake|correct)
                # 返回prob最大的candidates
                for candi in candidates:
                    prob = 0
                    # 计算logP（mistake|correct）
                    for candi in candidates:
                        prob = 0
                        # a. 计算log p(mistake|correct)
                        if candi in misspell_prob and word in misspell_prob[candi]:
                            prob += np.log(misspell_prob[candi][word])
                        else:
                            prob += np.log(0.0001)
                        # b. log p(correct)，计算计算过程中使用了Add-one Smoothing的平滑操作
                        # 先计算log p(word|pre_word)
                        pre_word = word_list[index - 1] if index > 0 else '<s>'
                        biagram_pre = ' '.join([pre_word, word])
                        if pre_word in term_count and biagram_pre in bigram_term_count:
                            prob += np.log((bigram_term_count[biagram_pre] + 1) / (term_count[pre_word] + V))
                        elif pre_word in term_count:
                            prob += np.log(1 / (term_count[pre_word] + V))
                        else:
                            prob += np.log(1 / V)
                        # 再计算log p(next_word|word)
                        if index + 1 < len(word_list):
                            next_word = word_list[index + 1]
                            biagram_next = ' '.join([word, next_word])
                            if word in term_count and biagram_next in bigram_term_count:
                                prob += np.log((bigram_term_count[biagram_next] + 1) / (term_count[word] + V))
                            elif word in term_count:
                                prob += np.log(1 / (term_count[word] + V))
                            else:
                                prob += np.log(1 / V)

                        probs.append(prob)
                        prob_dict[candi] = prob
                    if probs:
                        max_idx = probs.index(max(probs))
                        print(word, candidates[max_idx])
                        print(prob_dict)
                        s = input()
                    else:
                        print(word, False)



if __name__ == '__main__':
    res = read_file()
    print(res)

