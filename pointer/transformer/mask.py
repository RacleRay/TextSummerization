#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   mask.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn


def create_padding_mask(seq, pad_id):
    "注意，此处的mask为0表示有效内容，为1表示pad"
    mask = seq.eq(pad_id).byte()  # (batch, seq_len)
    if torch.cuda.is_available():
        mask = mask.type(torch.cuda.ByteTensor)
    length = seq.shape[1]
    mask = mask.unsqueeze(1).expand(-1, length, -1)  # does not allocate new memory
    mask = mask.repeat(n_head, 1, 1)  # (batch*n_head, length, length)
    return mask


def create_look_ahead_mask(maxlen):
     """比如，预测第三个词，将仅使用第一个和第二个词。mask未来。"""
     mask = torch.triu(torch.ones(maxlen, maxlen), dtype=torch.uint8, diagonal=1).byte()
     if torch.cuda.is_available():
        mask = mask.type(torch.cuda.ByteTensor)
    return mask


def create_decoder_mask(target_seq, pad_id):
    # encoder-decoder attention uses
    enc_mask = create_padding_mask(encoder_inp, pad_id)

    look_ahead_mask = create_look_ahead_mask(target_seq.shape[1])
    dec_target_padding_mask = create_padding_mask(target_seq)
    # decoder attention uses
    look_ahead_mask = (look_ahead_mask + dec_target_padding_mask).gt(0)  # broadcast
    return look_ahead_mask

