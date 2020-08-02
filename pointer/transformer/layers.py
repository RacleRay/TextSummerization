#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   layers.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import torch.nn as nn
from attention import MultiHeadAttention, PointWiseFeedForward
from utils import Embeddings
from mask import create_padding_mask


"""
NOTE: 有研究论文发现，将layer norm放在self attention之前和feed forward之前
能够加快收敛，减少对warm up的依赖。(pre norm)
"""

class EncoderLayer(nn.Module):
    def __init__(self,
                 n_head, d_model,
                 d_q, d_k, d_v,
                 d_affine,
                 dropout=0.1,
                 fc_dorpout=0.5):
        super.__init__()
        self.selfAttn = MultiHeadAttention(n_head, d_model, d_q, d_k, d_v, dropout)
        self.feedForward = PointWiseFeedForward(d_model, d_affine, fc_dorpout)

    def forward(self, x, mask=None):
        """
        x: (batch, q_len, d_model)
        mask: (batch, q_len, q_len)
        return:
            output: (batch, q_len, d_model)
            attention: (batch, q_len, q_len)
        """
        output, attention = self.selfAttn(x, x, x, mask)
        output = self.feedForward(output)
        return output, attention


class Encoder(nn.Module):
    def __init__(self,
                num_layers,
                n_head, d_model,
                vocab_size, max_seq_len,
                d_q, d_k, d_v,
                d_affine,
                embedding_dropout=0.1,
                dropout=0.1,
                fc_dorpout=0.5):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embeddings(d_model, vocab_size, max_seq_len, embedding_dropout)
        self.enc_layers = nn.ModuleList(
            [EncoderLayer(n_head, d_model, d_q, d_k, d_v, d_affine, dropout, fc_dorpout)
             for i in range(num_layers)]
        )

    def forward(self, x, pad_id):
        """
        x: 一个batch的id序列
        mask: pad标记mask， (batch*n_head, q_len, q_len)

        return:
            x: 最后一层输出
            layer_out_list：每一层的输出
            attn_list：每一层的multi-head self-attention
        """
        x = self.embedding(x)
        mask = create_padding_mask(x, pad_id)

        attn_list = []
        layer_out_list = []
        for i in range(self.num_layers):
            x, attention = self.enc_layers[i](x, mask)
            layer_out_list.append(x)  # (batch_size, seq_len, d_model)
            attn_list.append(attention)  # (batch, q_len, n_head, q_len)

        return x, layer_out_list, attn_list


class DecoderLayer(nn.Module):
    def __init__(self,
                 n_head, d_model,
                 d_q, d_k, d_v,
                 d_affine,
                 dropout=0.1,
                 fc_dorpout=0.5,
                 is_last=False):
        super()__init__()
        self.is_last = is_last
        self.dec_attn = MultiHeadAttention(n_head, d_model, d_v, d_v, d_v, dropout)
        self.enc_dec_attn = MultiHeadAttention(n_head, d_model, d_q, d_k, d_v, dropout, is_last)
        if not is_last:
            self.feedForward = PointWiseFeedForward(d_model, d_affine, fc_dorpout)

    def forward(self, x, enc_output, look_ahead_mask, enc_mask):
        dec_output, dec_attention = self.dec_attn(x, x, x, look_ahead_mask)
        output, dec_enc_attention = self.enc_dec_attn(dec_output,
                                                      enc_output,
                                                      enc_output)
        if not self.is_last:
            output = self.feedForward(output)
        return output, dec_attention, dec_enc_attention


class Decoder(nn.Module):
    def __init__(self,
                num_layers,
                n_head, d_model,
                vocab_size, max_seq_len,
                d_q, d_k, d_v,
                d_affine,
                embedding_dropout=0.1,
                dropout=0.1,
                fc_dorpout=0.5):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = Embeddings(d_model, vocab_size, max_seq_len, embedding_dropout)
        self.dec_layers = nn.ModuleList(
            [DecoderLayer(n_head, d_model, d_q, d_k, d_v, d_affine, dropout, fc_dorpout)
             for i in range(num_layers - 1)]
        )
        self.last_layers = DecoderLayer(n_head, d_model, d_q, d_k, d_v,
            d_affine, dropout, fc_dorpout, is_last=True)

    def forward(self, x, enc_output, look_ahead_mask, enc_mask):
        """
        enc_output: encoder最后一层的输出

        return:
            x: 最后一层输出
            layer_out_list：每一层的输出
            attn_list：每一层的multi-head self-attention
        """
        x = self.embedding(x)
        attn_list = []
        layer_out_list = []
        for i in range(num_layers - 1):
            x, attention = self.dec_layers[i](x, enc_output, look_ahead_mask, enc_mask)
            layer_out_list.append(x)  # (batch_size, seq_len, d_model)
            attn_list.append(attention)  # (batch, q_len, n_head, q_len)
        x, attention = self.last_layers(x, enc_output, look_ahead_mask, enc_mask)
        layer_out_list.append(x)
        attn_list.append(attention)
        return x, layer_out_list, attn_list