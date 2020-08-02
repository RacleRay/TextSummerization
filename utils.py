#-*- coding:utf-8 -*-
# author: Racle
# project: autosummarization

import re
from pyltp import SentenceSplitter
from pyhanlp import *
from gensim.models import KeyedVectors
import pickle

tokenizer = JClass('com.hankcs.hanlp.tokenizer.StandardTokenizer')

def get_frequence(path):
    with open(path, 'rb') as f:
        frequence = pickle.load(f, encoding='uft-8')
    return frequence


def get_stopwords(path):
    """创建停用词列表"""
    stopwords = {line.strip()
        for line in open(path, encoding='UTF-8').readlines()}
    stopwords.add('\u3000')
    return stopwords


def get_word_vec(path):
    word_vec = KeyedVectors.load_word2vec_format(path)
    return word_vec


def split_to_sentence(doc, min_len=6, use_re=False):
    """自定义的分段分句

    return:
        list, 储存内容为每一段分句结果的list，index信息可以用于后续位置特征计算
    """
    if use_re:
        pattern = re.compile(".*?[。?？!！]")  # 非贪婪模式匹配文字内容

    paragraph_gen = split_to_paragraph(doc)
    doc_content = []
    for para in paragraph_gen:
        if para is None:
            continue
        elif len(para) <= min_len:
            continue

        if not use_re:
            doc_content.append(split_sentence(para))
        else:
            if para.strip()[-1] in '。?？!！"”':
                sent_of_para = re.findall(pattern, para)
                doc_content.append(sent_of_para)
            else:
                doc_content.append([para])

    return doc_content


def split_to_paragraph(doc):
    """为了识别靠近段开头和结尾位置，需要单独输出句子位置特征

    return:
        filter结果生成器
    """
    pattern = re.compile(r"(\r\n\u3000\u3000)|(\r\n)|(\u3000\u3000)|(\\n)")
    res = re.split(pattern, doc)
    for i in res:
        if i and len(i) > 5:
            yield i


def split_sentence(para):
    return [sent for sent in SentenceSplitter.split(para) if len(sent)>5]


def get_position_weight(sent_para):
    """开头，结尾位置增加一些权重

    input:
        分段分句的结果
    return:
        从第一句到最后一句的位置权重，list
    """
    pos_sent_weight = []
    first_para_flag = True
    for i, para in enumerate(sent_para):
        if len(para) > 1:
            # 每一段开头结尾
            tmp = [1.1] + [1. for i in range(len(para)-2)] + [1.08]
        else:
            tmp = [1.]
        # 第一段
        if first_para_flag:
            tmp = [1.1 * i for i in tmp]
            first_para_flag = False
        # 最后一段
        elif i == len(sent_para) - 1 and len(para[-1]) >= 10:
            tmp = [1.08 * i for i in tmp]
        pos_sent_weight.extend(tmp)
    return pos_sent_weight


