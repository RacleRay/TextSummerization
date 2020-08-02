#-*- coding:utf-8 -*-
# author: Racle
# project: autosummarization

import numpy as np
from scipy.spatial.distance import cosine
from gensim import corpora, models
from itertools import chain
from textrank import TextRank
from utils import *
from config import CONFIG


class EmbeddingModel:
    def __init__(self):
        self.word_vec = get_word_vec(CONFIG.wordVecPath)
        self.lda = models.LdaModel.load(CONFIG.ldaModelPath)
        self.dictionary = corpora.Dictionary.load(CONFIG.ldaDictPath)

        self.frequence = get_frequence(CONFIG.wordFreqPath)
        self.stopwords = get_stopwords(CONFIG.stopWordsPath)

        self.num_topics = CONFIG.numTopics

        self.debug = CONFIG.debug


    def summary(self, doc, title=None, use_textrank_keysent=False):
        """输出文本摘要和关键词。"""
        # 处理输入
        sent_para = split_to_sentence(doc)
        self.sent_list = [sent.strip() for sent in chain.from_iterable(sent_para)]
        sent_num = len(self.sent_list)

        with_title = False
        if title:
            self.sent_list.append(title)
            with_title = True

        pos_sent_weight = get_position_weight(sent_para)
        del sent_para

        # embedding计算
        sent_vecs, title_vec, doc_vec, total_tokens = self.__cal_sentences_vec_mat(with_title)

        # 由于lda从文档中抽象出topic实际上时对语义信息的另一种建模，不加入sentence embedding算法实现
        topic_dist, topic_words_dist = self.get_topic_distribution(total_tokens)
        topics_vec = self.__cal_topic_embedding(topic_dist, topic_words_dist)

        # keyword
        textrank = TextRank()
        self.keywords = textrank.get_keywords(doc)

        # 计算得分
        scores = self.__cal_score(sent_vecs,
                                  doc_vec,
                                  topics_vec,
                                  title_vec,
                                  pos_sent_weight,
                                  sent_num)
        score_smooth = self.__score_smooth(scores, sent_num)

        # 排序
        sorted_idx = np.argsort(score_smooth)[-sent_num//3: ]
        sent_ids = sorted(sorted_idx)

        if self.debug:
            print('key words: ', self.keywords)
            print('position weight: ', pos_sent_weight)
            print('score:', scores)
            print('score smooth:', score_smooth)
            for i in sent_ids:
                print(self.sent_list[i])

        if use_textrank_keysent:
            keysentence = textrank.get_keysentences(doc)
            print('textrank keysentence: ', keysentence)

        return ''.join([self.sent_list[i] for i in sent_ids]), \
               '；'.join([w for w, _ in self.keywords])

    def __cal_score(self, sent_vecs, doc_vec, topics_vec, title_vec, pos_sent_weight, sent_num):
        """计算每个句子的得分，约束到[0, 2]范围的cosine similarity score。"""
        scores = []
        kw, values = zip(*self.keywords)
        max_v = max(values)
        min_v = min(values)
        values = [(v - min_v) / (max_v - min_v) for v in values]
        # values = [v / max_v for v in values]
        for i in range(sent_num):
            # cosine计算cosine distance = 1 - cosine similarity。此处得分越高越好，同时将其约束到[0，2]之间
            sent_to_doc = (2 - cosine(sent_vecs[:, i], doc_vec)) * pos_sent_weight[i]
            sent_to_topic = (2 - cosine(sent_vecs[:, i], topics_vec))
            if title_vec:
                sent_to_title = (2 - cosine(sent_vecs[:, i], title_vec))
                score = sent_to_doc + sent_to_topic + sent_to_title
            else:
                score = sent_to_doc + sent_to_topic

            for (w, v) in zip(kw, values):
                if w in self.sent_list[i]:
                    # 根据value大小顺序，递减权重
                    score *= (1 + 0.5 * v)

            scores.append(score)
        return scores

    def __score_smooth(self, scores, sent_num):
        # 对于一个sentence，它的重要性，取决于本身的重要性和周围的句子(neighbors)的重要性的综合
        window = 1  # smooth的window大小，注意和weight的shape一起调整
        for i in range(window):
            scores.insert(0, scores[0])
            scores.append(scores[-1])
        weight = np.array([0.25, 0.5, 0.25])

        scores = np.array(scores)
        score_smooth = [np.dot(scores[i - window: i + window + 1], weight) \
                        for i in range(window, sent_num + window)]

        assert sent_num == len(score_smooth)
        return score_smooth

    def __cal_sentences_vec_mat(self, with_title, param_a=CONFIG.embedModelParaA):
        """计算sentence vector，在原论文的基础上进行修改，语义建模引入整个文档和标题信息。
        基本思想来自paper:
        A SIMPLE  BUTTOUGH-TO-BEATBASELINE  FORSEN-TENCEEMBEDDINGS. ICLR2017

        sent_list: 来自待识别文档的分句结果, 最后一维为标题信息，list；
        param_a: 论文中实验得到的效果比较好的参数取值, 1e-3 ~ 1e-5；

        return:
            matrix--(vector_dim, sentence_num + 1)
                    形状的matrix，每一列代表sentence的向量. 多出的1为doc的向量.
                    如果有title，sentence_num中最后一个sent为title
            title_vector -- title的向量表达
            doc_vector--整个文档的向量表达
        """
        row_size = self.word_vec.vector_size
        col_size = len(self.sent_list)

        doc_vector = np.zeros(row_size)
        matrix = np.zeros((row_size, col_size + 1))  # +1为整个文档的向量表示
        total_tokens = []  # 计算文档主题分布时使用

        default_p = max(self.frequence.values())
        doc_len = 0
        for i, sentence in enumerate(self.sent_list):
            sentence = tokenizer.segment(sentence)
            sent_len = len(sentence)
            doc_len += 1
            sent_vector = matrix[:, i]
            for item in sentence:  # 计算第i句的sent_vector
                token = str(item.word)
                if token not in self.stopwords:
                    total_tokens.append(token)
                pw = self.frequence.setdefault(token, default_p)
                weight = param_a / (param_a + pw)
                try:
                    word_vector = np.array(self.word_vec.get_vector(token))
                    sent_vector += weight * word_vector
                except Exception:
                    continue

            matrix[:, i] = sent_vector / sent_len
            doc_vector += matrix[:, i]
        matrix[:, -1] = doc_vector / doc_len

        # print(matrix)
        matrix = np.nan_to_num(matrix)
        # PCA找到整个矩阵中，每个句子中最相似的部分（第一个主成分），然后减去相似部分
        U, s, Vh = np.linalg.svd(matrix)  # 默认s降序
        u = U[:, 0]  # 第一个主成分
        matrix -= np.outer(u, u.T) @ matrix  # 每个sent_vector减去在第一个主成分方向的投影

        doc_vector = matrix[:, -1]
        title_vector = None
        if with_title:
            title_vector = matrix[:, -2]
        return matrix, title_vector, doc_vector, total_tokens

    def get_topic_distribution(self, total_tokens):
        """用每句话和的主题进行相似度对比。使用LDA主题模型，得到的主题分布。

        return:
            topic_dist：
                format--[(1, 0.018213129),
                        (2, 0.06460305),
                        (3, 0.114253126),
                        (5, 0.21796304),
                        (6, 0.03961128),
                        (9, 0.5442903)]
            topic_words_dist:
                [[topic, one, words],
                 [topic, two, words]...]
        """
        bow_doc = self.dictionary.doc2bow(total_tokens)
        topic_dist = self.lda.get_document_topics(bow_doc)

        topic_words_dist = []  # 每个主题分布中前十的word来表示该主题
        for topicid in range(self.num_topics):
            topic_words = [w for w, _ in self.lda.show_topic(topicid, topn=10)]
            topic_words_dist.append(topic_words)

        # 根据主题分布，和每个主题中word的分布，获得需要的主题词的分布
        return topic_dist, topic_words_dist

    def __cal_topic_embedding(self, topic_dist, topic_words_dist, param_a=CONFIG.embedModelParaA):
        """根据主题分布，每个主题的词分布，获取topic embedding。

        return:
            vector_size大小的一维vector
        """
        size = self.word_vec.vector_size
        default_p = max(self.frequence.values())
        topics_vector = np.zeros(size)
        while topic_dist:
            # topic weight加权
            topicid, t_weight = topic_dist.pop()
            topic_words = topic_words_dist[topicid]
            # 与计算sentence embedding的方法保持一致
            topic_vector = np.zeros(size)
            for word in topic_words:
                pw = self.frequence.setdefault(word, default_p)
                w_weight = param_a / (param_a + pw)
                try:
                    word_vector = np.array(self.word_vec.get_vector(word))
                    topic_vector += w_weight * word_vector
                except Exception:
                    continue
            topics_vector += t_weight * topic_vector
        topics_vector /= self.num_topics * 10  # 每个topic选取10个词来表示
        return topics_vector