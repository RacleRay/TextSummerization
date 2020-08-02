#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   attention.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import numpy as np
import torch
import torch.nn as nn


class LuongAttention(nn.Module):
    def __init__(self, hidden_size, attn_dropout=0.1):
        super().__init__()
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, hidden):
        """
        :param query: (batch, 1,     hidden_size) decoder output
        :param hidden: (batch, t_len, hidden_size) encoder hidden state
        """
        query = self.linear_q(query)
        query = query.transpose(1, 2)
        attention = nn.Softmax(dim=-1)(torch.bmm(hidden, query))
        attention_t = attention.transpose(1, 2)
        if self.attn_dropout != 0:
            attention_t = self.dropout(attention_t)
        context = torch.bmm(attention_t, hidden)
        # (batch, t_len)  (batch, 1, hidden_size)
        return attention.squeeze(), context


class ScaledDotProductAtt(nn.Module):
    def __init__(self, attn_dropout=0.1):
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        q, k, v: (batch*n_head, t_len, hidden_size)
        mask: (batch*n_head, q_len, q_len)
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        attention = attention / torch.sqrt(torch.cast(q.shape[-1], torch.float32))
        if mask is not None:
            attention = attention.masked_fill(mask, -np.inf)
        attention = self.softmax(attention)  # (batch*n_head, t_len, t_len)
        output = torch.bmm(attention, v)     # (batch*n_head, t_len, hidden_size)
        return output, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_q, d_k, d_v, dropout=0.1):
        """一般 n_head * d_q = d_model; d_q/d_k/d_v 是相同的"""
        super().__init__()
        self.n_head = n_head
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v

        # 层与层之间全连接, 计算输入q/k/v
        self.w_q = nn.Linear(d_model, n_head * d_q)
        self.w_k = nn.Linear(d_model, n_head * d_k)
        self.w_v = nn.Linear(d_model, n_head * d_v)

        # Xavier(Glorot)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAtt()

        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, is_last=False):
        """
        mask: (batch*n_head, q_len, k_len)
        is_last: 最后一层的decoder只输出到第二个multi head attention之后
        """
        d_q, d_k, d_v, n_head = self.d_q, self.d_k, self.d_v, self.n_head
        batch, q_len, _ = q.size()
        _, k_len, _ = k.size()
        _, v_len, _ = v.size()
        residual = q

        q = self.w_q(q).view(batch, q_len, n_head, d_q)
        k = self.w_k(k).view(batch, k_len, n_head, d_k)
        v = self.w_v(v).view(batch, v_len, n_head, d_v)

        # (batch*n_head, q_len, d_q)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, q_len, d_q)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, k_len, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, v_len, d_v)

        output, attention = self.attention(q, k, v, mask=mask)
        attention = attention.view(batch, n_head, q_len, d_v)
        attention = attention.premute(0, 2, 1, 3).contiguous()  # (batch, q_len, n_head, k_len)
        output = output.view(n_head, batch, q_len, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch, q_len, -1)

        if is_last:
            return output, attention

        output = self.dropout(self.activation(self.fc(output)))
        output = self.layer_norm(output + residual)
        # (batch, q_len, d_v*n_head)  (batch, q_len, n_head, q_len)
        return output, attention


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_affine, fc_dorpout=0.2):
        super().__init__()
        self.d_model = d_model
        self.d_affine = d_affine

        self.linear_1 = nn.Linear(self.d_model, self.d_affine)
        self.linear_2 = nn.Linear(self.d_affine, self.d_model)
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout_1 = nn.Dropout(fc_dorpout)
        self.dropout_2 = nn.Dropout(fc_dorpout)
        self.selu = nn.SELU()

    def forward(self, x):
        output = self.dropout_1(self.selu(self.linear_1(self.layer_norm(x))))
        output = self.dropout_2(self.linear_2(output))
        return output + x