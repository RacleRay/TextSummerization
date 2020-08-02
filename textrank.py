#-*- coding:utf-8 -*-
# author: Racle
# project: autosummarization


# coding:utf-8

import numpy as np

from collections import defaultdict, Counter
from itertools import chain
from math import log
from utils import *
from config import CONFIG

np.seterr(divide='ignore', invalid='ignore')


class TextRank:
    def __init__(self):
        """实现textrank keywords抽取，textrank keysentences抽取，以及矩阵并行的pagerank。
        keywords抽取效果和开源库jieba、hanlp相当。keysentences抽取结果不如基于embedding + keywords加权的方法。
        keysentences抽取对于较长的句子抽取较难，需要对BM25中参数k、b进行调整。实际效果并不理想。
        对比hanlp中的keysentences抽取，效果相对，但是效果都一般。
        """
        self.stopwords = get_stopwords(CONFIG.stopWordsPath)
        self.allowed_pos = frozenset(CONFIG.allowPosTags)

    def process_input(self, doc_str, case='keyword'):
        "处理输入文档。输出结果格式为：[['sent', 'one', 'words'],['sent', 'two', 'words']]"
        self.sent_of_words = []
        sent_list = split_to_sentence(doc_str)
        for sent in chain.from_iterable(sent_list):
            tmp = []
            for item in tokenizer.segment(sent):
                if case == 'keyword':
                    tmp.append(item)
                elif case == 'keysentence':
                    tmp.append(str(item.word))
            self.sent_of_words.append(tmp)

    def get_keywords(self, doc_str, num=10, min_len=2, span=5):
        """

        :param span: TextRank共现窗口大小
        :return:
            [(keyword, value)]
        """
        self.process_input(doc_str, case='keyword')
        # 找到候选
        count_dic = defaultdict(int)
        word_set = set()
        for sent in self.sent_of_words:
            for i, w_item in enumerate(sent):
                if self.filte_words(w_item, min_len):
                    word = str(w_item.word)
                    word_set.add(word)
                    for j in range(i+1, i+span):
                        if j >= len(sent): break
                        if not self.filte_words(sent[j], min_len):
                            continue
                        word_j = str(sent[j].word)
                        word_set.add(word_j)
                        count_dic[(word, word_j)] += 1
                        count_dic[(word_j, word)] += 1
        word2id = {word: i for i, word in enumerate(list(word_set))}
        id2word = {i: word for word, i in word2id.items()}

        # 建立共线矩阵
        num_of_nodes = len(word2id)
        weight_M = np.zeros((num_of_nodes, num_of_nodes))
        for (wi, wj), weight in count_dic.items():
            i = word2id[wi]
            j = word2id[wj]
            weight_M[i, j] = weight

        weight_M = np.nan_to_num(weight_M / np.linalg.norm(weight_M,
                                                           ord=1,
                                                           axis=0,
                                                           keepdims=True))
        # pagerank求解
        textrank_v = self.page_rank(weight_M)
        result = sorted([(id2word[i], value) for i, value in enumerate(textrank_v)],
                        key=lambda x: x[1],
                        reverse=True)
        return result[: num]

    def get_keysentences(self, doc_str, num=6, min_len=5):
        """由于sentence在一段话中几乎不可能出现完全一样的情况，因此只基于共现的pagerank是行不通的。
        引入BM25，来计算句子与句子之间的关联权重。注：BM25原本是用来计算query句子和文档之间的相似度，用于信息检索的、

        :return:
            [(keysentence, value， sentence index)]
        """
        self.process_input(doc_str, case='keysentence')
        total_sent = len(self.sent_of_words)

        # 计算权重
        weight_M = np.zeros((total_sent, total_sent))
        for i in range(total_sent):
            sent_i = self.sent_of_words[i]
            for j in range(total_sent):
                if i == j: continue
                sent_j = self.sent_of_words[j]
                # 权重矩阵中的i行，j列
                weight_M[i, j] = self.sent_corelation_func(sent_i, sent_j)
        weight_M = np.nan_to_num(weight_M / np.linalg.norm(weight_M,
                                                           ord=1,
                                                           axis=0,
                                                           keepdims=True))
        # 计算每一句的重要性得分
        sent_para = split_to_sentence(doc_str)
        ps_weight = get_position_weight(sent_para)
        textrank_v = self.page_rank(weight_M) * np.array(ps_weight)

        result_id = sorted([(idx, value) for idx, value in enumerate(textrank_v)],
                           key=lambda x: x[1],
                           reverse=True
                           )

        # 处理输出结果
        count = 0
        result_sent = []
        for (i, value) in result_id:
            if count >= num:
                break
            sent = ''.join(self.sent_of_words[i])
            if len(sent) <= min_len:
                continue
            result_sent.append((sent, value, i))
            count += 1

        result_sent = sorted(result_sent, key=lambda x: x[2])
        return result_sent


    def get_tf(self, sent_i, sent_j):
        """计算bm25的term frequence. sent来自预处理的sent_of_words列表。"""
        freq = {}
        sent_i_counts = Counter(sent_i)
        # 计算i句中的词，在j句中的tf
        for w in sent_j:
            # if not self.filte_words(w_item):
            #     continue
            if w in sent_i_counts:
                freq[w] = sent_i_counts[w]
            else:
                freq[w] = 0
        total = len(sent_i)
        return {word: count / total for word, count in freq.items()}

    def get_idf(self):
        """计算inverse document frequence. 这里计算句子的相似度，所以计算inverse sentence frequence"""
        total_sent = len(self.sent_of_words) + 1 # 假设有一个句子包含所有词
        avg_len = 0
        doc_freq = {}
        for sent in self.sent_of_words:
            avg_len += len(sent)
            words = list({w for w in sent})
            for word in words:
                # 假设有一个句子包含所有词
                doc_freq[word] = doc_freq.setdefault(word, 1) + 1
        avg_len /= total_sent
        # sklearn中的实现方式
        idf = {word: log(total_sent / df) + 1 for word, df in doc_freq.items()}
        return idf, avg_len

    def filte_words(self, w_item, min_len=2):
        word = str(w_item.word)
        pos = str(w_item.nature)
        return (pos in self.allowed_pos and word not in self.stopwords
                and len(word) >= min_len)

    def sent_corelation_func(self, sent_i, sent_j, k1=1.5, b=0.75):
        """计算bm25。

        sent_i ： 与query对比的句子，在文档中进行遍历
        sent_j : query的句子
        """
        idf, avg_len = self.get_idf()
        tf = self.get_tf(sent_i, sent_j)

        K = k1 * (1 - b + b * len(sent_i) / avg_len)
        bm25 = 0
        for j_word in sent_j:
            bm25 += idf[j_word] * tf[j_word] * (k1 + 1) / (tf[j_word] + K)
        return bm25

    @staticmethod
    def page_rank(weight_M, iterations=100, d=0.85):
        """
        weight_M: 对于textRank，这是窗口遍历文档所得的符合条件的边的权重矩阵。
                  pageRank中第i行、第j列表示：从j节点到i节点的链接权重。
                  但是textRank是无向图，只是两者的共性关系权重。
        d： 衰减系数，防止局部陷入无法向外链接
        """
        N = weight_M.shape[1]
        v = np.random.rand(N, 1)
        v = v / np.linalg.norm(v, 1)
        M_hat = (d * weight_M + (1 - d) / N)
        for i in range(iterations):
            v = M_hat @ v
        return v.ravel()

