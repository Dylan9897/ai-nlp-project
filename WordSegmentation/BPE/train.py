# encoding : utf-8 -*-                            
# @author  : 冬瓜                              
# @mail    : dylan_han@126.com    
# @Time    : 2025/1/23 13:44
import re
import collections

def get_vocab(filepath):
    """ 读取文件，并生成词表，每个单词按字符分隔，并加上结束符 :param filepath: :return: """
    vocab = collections.defaultdict(int)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.strip("\ufeff").strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab

def measure_token_length(token):
    """ 获取token的长度 :param token: :return: """
    if token[-4:] == '</w>':
        return len(token[:-4]) + 1
    else:
        return len(token)

class BpeObject():
    def __init__(self):
        pass

    def get_stats(self, vocab):
        """ 统计两两组合的字符 pairs 出现的频率 :param vocab: :return: """
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    # def get_tokens(self, vocab):     # """     # 统计 vocab 中每个字符的频率     # :param vocab:     # :return:     # """     # tokens = collections.defaultdict(int)     # for word, freq in vocab.items():     # word_tokens = word.split()     # for token in word_tokens:     # tokens[token] += freq     # return tokens
    def merge_vocab(self, pair, v_in):
        """ 将频率最高的字符串$(A,B)$加入到词库中，将语料中所有的$(A,B)$替换成$AB$。 :param pair: 频率最高的字符串 :param v_in: 词表 :return: """
        v_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def get_tokens_from_vocab(self, vocab, verbose=False):
        """ 统计 vocab 中每个字符的频率 :param vocab: :return: """
        tokens_frequencies = collections.defaultdict(int)
        vocab_tokenization = {}
        for word, freq in vocab.items():
            if verbose:
                print(f"checkpoint for word is {word}")
                print(f"checkpoinf for freq is {freq}")
            word_tokens = word.split()
            for token in word_tokens:
                tokens_frequencies[token] += freq
            vocab_tokenization[''.join(word_tokens)] = word_tokens
        return tokens_frequencies, vocab_tokenization

    def tokenize(self, string, sorted_tokens, unknown_token="</u>"):
        """ 递归算法：对单词进行切分 :param string: :param sorted_tokens: :param unknown_token: :return: """
        if string == "":
            return []
        if sorted_tokens == []:
            return unknown_token
        string_tokens = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_reg = re.escape(token.replace('.', '[.]'))

            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
            if len(matched_positions) == 0:
                continue

            substring_end_positions = [matched_position[0] for matched_position in matched_positions]

            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += self.tokenize(string=substring, sorted_tokens=sorted_tokens[i + 1:],
                                                    unknown_token=unknown_token)
                string_tokens += [token]
                substring_start_position = substring_end_position + len(token)
            remaining_substring = string[substring_start_position:]
            string_tokens += self.tokenize(string=remaining_substring, sorted_tokens=sorted_tokens[i + 1:],
                                                unknown_token=unknown_token)
            break
        return string_tokens


if __name__ == '__main__':
    from tqdm import tqdm
    # 读取文件
    vocab = get_vocab("pg16457.txt")
    # 初始化 bpe 函数
    bpe = BpeObject()

    tokens_frequencies, vocab_tokenization = bpe.get_tokens_from_vocab(vocab)

    print('All tokens: {}'.format(tokens_frequencies.keys()))
    print('Number of tokens: {}'.format(len(tokens_frequencies.keys())))
    print('==========')

    num_merges = 100
    for i in tqdm(range(num_merges)):
        pairs = bpe.get_stats(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = bpe.merge_vocab(best, vocab)
        tokens_frequencies, vocab_tokenization = bpe.get_tokens_from_vocab(vocab)

    test_word_list = ['mountains</w>','Ilikeeatingapples!</w>']

    # 对生成的词表进行排序
    sorted_tokens_tuple = sorted(tokens_frequencies.items(), key=lambda item: (measure_token_length(item[0]), item[1]),
                                 reverse=True)
    sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]
    for elem in test_word_list:
        if elem in vocab_tokenization:
            print(f"after tokenizer, result is {vocab_tokenization[elem]}")
        else:
            word = bpe.tokenize(string=elem, sorted_tokens=sorted_tokens, unknown_token='</u>')
            print(f"after tokenizer, result is {word}")