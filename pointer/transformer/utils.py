#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''


import math
import numpy as np
import matplotlib.pyplot as plt
import os
import pyrouge
import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
import tensorflow as tf
from mask import create_decoder_mask, create_padding_mask

import config


def gen_decoder_input(string, bosIdx):
    "增加<bos>, 输入向左shift一位"
    if torch.cuda.is_available():
        longType = torch.cuda.LongTensor
    else:
        longType = torch.LongTensor
    bos = (torch.ones(string.size(0), 1) * bosIdx).type(longType)
    string = torch.cat((bos, string), dim=1)[:, :-1]
    return string


def get_angles(pos, i, d_model):
    """pos -- word id; i -- embedding dim id; d_model -- embedding dim"""
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(sentence_length, d_model):
    angle_rads = get_angles(
        np.arange(sentence_length)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :], d_model)
    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    pos_encoding = torch.cast(pos_encoding, dtype=torch.float32)
    pos_encoding = pos_encoding.permute(1, 0, 2).contiguous()  # (len, 1, d_model)
    return pos_encoding


def plot_position_embedding(sentence_length, d_model):
    pos_encoding = positional_encoding(sentence_length, d_model)
    print(pos_encoding.shape)
    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 512))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size, max_seq_len=500, dropout=0.1):
        super.__init__()
        self.d_model = d_model
        self.pos_embedding = nn.Embeddings.from_pretrained(
            positional_encoding(max_seq_len, d_model), freeze=True
        )
        self.word_embedding = nn.Embeddings(vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

    def forward(self, x):
        word_embedding = self.word_embedding(x)
        # 保证x转化到 d ^ (1/2) 正态分布，且要比position encoding大
        word_embedding *= torch.sqrt(torch.cast(self.d_model, torch.float32))
        # (batch, seq_len)
        pos_seq = torch.arange(0, x.size(1)).repeat(x.size(0), 1)
        if torch.cuda.is_available():
            pos_seq = pos_seq.cuda()
        embedding = word_embedding + self.pos_embedding(pos_seq)
        embedding = self.embedding_dropout(embedding)
        return embedding, pos_seq


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.bias.data.zero_()
        self.weight.data.fill_(1.0)

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


def get_encoder_variables(batch, use_cuda):
    "创建encoder所需变量"
    batch_size = len(batch.enc_lens)
    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_lens = batch.enc_lens
    extra_zeros = None
    enc_batch_extend_vocab = None

    if config.pointer_gen:
        enc_batch_extend_vocab = Variable(torch.from_numpy(batch.enc_batch_extend_vocab).long())
        # max_art_oovs is the max over all the article oov list in the batch
        if batch.max_art_oovs > 0:
            extra_zeros = Variable(torch.zeros((batch_size, batch.max_art_oovs)))

    context_v_1 = Variable(torch.zeros((batch_size, config.d_model)))

    coverage = None
    if config.do_coverage:
        coverage = Variable(torch.zeros(enc_batch.size()))

    if use_cuda:
        enc_batch = enc_batch.cuda()

        if enc_batch_extend_vocab is not None:
            enc_batch_extend_vocab = enc_batch_extend_vocab.cuda()
        if extra_zeros is not None:
            extra_zeros = extra_zeros.cuda()
        context_v_1 = context_v_1.cuda()

        if coverage is not None:
            coverage = coverage.cuda()

    return enc_batch, enc_batch_extend_vocab, enc_lens, extra_zeros, context_v_1, coverage


def get_decoder_variables(batch, use_cuda):
    "创建decoder所需的变量"
    dec_batch = Variable(torch.from_numpy(batch.dec_batch).long())
    target_batch = Variable(torch.from_numpy(batch.target_batch)).long()
    enc_batch = Variable(torch.from_numpy(batch.enc_batch).long())
    enc_mask = create_padding_mask(enc_batch, config.pad_id)
    look_ahead_mask = create_decoder_mask(dec_batch, target_batch, config.pad_id)

    dec_lens = batch.dec_lens
    dec_lens_var = Variable(torch.from_numpy(dec_lens)).float()

    if use_cuda:
        dec_batch = dec_batch.cuda()
        target_batch = target_batch.cuda()
        enc_mask = enc_mask.cuda()
        look_ahead_mask = look_ahead_mask.cuda()
        dec_lens_var = dec_lens_var.cuda()

    return dec_batch, target_batch, enc_mask, look_ahead_mask, dec_lens_var


def calc_moving_avg_loss(loss, moving_avg_loss, summary_writer, step, decay=0.99):
    "moving average loss"
    if moving_avg_loss == 0:  # on the first iteration just take the loss
        moving_avg_loss = loss
    else:
        moving_avg_loss = moving_avg_loss * decay + (1 - decay) * loss
    moving_avg_loss = min(moving_avg_loss, 12)  # clip

    loss_sum = tf.Summary()
    tag_name = 'running_avg_loss/decay=%f' % (decay)
    loss_sum.value.add(tag=tag_name, simple_value=moving_avg_loss)
    summary_writer.add_summary(loss_sum, step)

    return moving_avg_loss


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # doesn't correspond to an in-article oov
                raise ValueError(
                    'Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                        i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def rouge_eval(ref_dir, dec_dir):
    r = pyrouge.Rouge155()
    r.model_filename_pattern = '#ID#_reference.txt'
    r.system_filename_pattern = '(\d+)_decoded.txt'
    r.model_dir = ref_dir
    r.system_dir = dec_dir
    logging.getLogger('global').setLevel(logging.WARNING)  # silence pyrouge logging
    rouge_results = r.convert_and_evaluate()
    return r.output_to_dict(rouge_results)


def rouge_log(results_dict, dir_to_write):
    log_str = ""
    for x in ["1", "2", "l"]:
        log_str += "\nROUGE-%s:\n" % x
        for y in ["f_score", "recall", "precision"]:
            key = "rouge_%s_%s" % (x, y)
            key_cb = key + "_cb"
            key_ce = key + "_ce"
            val = results_dict[key]
            val_cb = results_dict[key_cb]
            val_ce = results_dict[key_ce]
            log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
    print(log_str)
    results_file = os.path.join(dir_to_write, "ROUGE_results.txt")
    print("Writing final ROUGE results to %s..." % (results_file))
    with open(results_file, "w") as f:
        f.write(log_str)


def write_for_rouge(reference_sents, decoded_words, ex_index,
                    _rouge_ref_dir, _rouge_dec_dir):
    decoded_sents = []
    while len(decoded_words) > 0:
        try:
            fst_period_idx = decoded_words.index(".")
        except ValueError:
            fst_period_idx = len(decoded_words)
        sent = decoded_words[:fst_period_idx + 1]
        decoded_words = decoded_words[fst_period_idx + 1:]
        decoded_sents.append(' '.join(sent))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    ref_file = os.path.join(_rouge_ref_dir, "%06d_reference.txt" % ex_index)
    decoded_file = os.path.join(_rouge_dec_dir, "%06d_decoded.txt" % ex_index)

    with open(ref_file, "w") as f:
        for idx, sent in enumerate(reference_sents):
            f.write(sent) if idx == len(reference_sents) - 1 else f.write(sent + "\n")
    with open(decoded_file, "w") as f:
        for idx, sent in enumerate(decoded_sents):
            f.write(sent) if idx == len(decoded_sents) - 1 else f.write(sent + "\n")

    # print("Wrote example %i to file" % ex_index)


def make_html_safe(s):
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s
