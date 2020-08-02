# -*- coding:utf-8 -*-
# author: Racle
# project: autosummarization

from itertools import chain
from utils import tokenizer, split_to_sentence
from gensim.models import FastText
from tqdm import tqdm
from collections import Counter
import pickle


def segment_for_fasttext(content):
    """在split_to_sentence的基础上，分词并按空格隔开。采用hanlp的StandardTokenizer。处理一个文本。
    处理时用pandas读取数据，然后对每一个文本的行apply该函数。保存为txt，一行为一个文本。

    """
    total_tokens = []
    sents = split_to_sentence(content)
    for sent in chain.from_iterable(sents):
        tokens = [str(item.word) for item in tokenizer.segment(sent)]
        total_tokens.extend(tokens)
    return ' '.join(total_tokens)


def fasttext_train(corpus_path, save_path):
    """输入分词完成的txt文件，一行为一个文本。"""
    model = FastText(window=5, size=200, min_count=1, workers=2)
    model.build_vocab(corpus_file=corpus_path)  # scan over corpus to build the vocabulary

    total_words = model.corpus_total_words  # number of words in the corpus
    model.train(corpus_file=corpus_path, total_words=total_words, epochs=5)

    model.save(save_path)


def count_gen(tokens):
    corpus_dict = []
    for c in tqdm(tokens):
        corpus_dict.extend(c.split())
    return corpus_dict


def cal_frequency(corpus):
    """

    :param corpus: 整个预料分词得到的iterable对象
    :return:
    """
    total_counter = Counter(corpus)
    length = len(corpus)
    frequence = {w: c/length for w, c in total_counter.items()}

    with open('frequence.bin', 'wb') as f:
        pickle.dump(frequence, f)


def segment_for_lda(sentences, stopwords):
    """在split_to_sentence的基础上，分词并去停用词。采用hanlp的StandardTokenizer。

    return:
        list of tokens for a doc.
    """
    total_tokens = []
    for sent in chain.from_iterable(sentences):
        tokens = [item.word for item in tokenizer.segment(sent) \
                  if item.word not in stopwords]
        total_tokens.extend(tokens)
    return total_tokens