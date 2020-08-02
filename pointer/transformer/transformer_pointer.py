#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   transformer_pointer.py
@Author  :   Racle
@Version :   1.0
@Desc    :   None
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import Encoder, Decoder
import config

import random

use_cuda = config.use_gpu and torch.cuda.is_available()

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


d_q = config.d_model // config.n_head
d_k = d_q
d_v = d_q


class PointerEncoder(nn.Module):
    def __init__(self,):
        super(PointerEncoder, self).__init__()
        self.encoder = Encoder(config.num_layers, config.n_head, config.d_model,
                               config.vocab_size, config.max_enc_len,
                               d_q, d_k, d_v, config.d_affine,
                               config.embedding_dropout, config.att_dropout, config.fc_dorpout)
        self.W_h = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, inputs):
        enc_output, _, _ = self.encoder(inputs)
        enc_feature = self.W_h(enc_output)
        return enc_output, enc_feature


class PointerAttention(nn.Module):
    def __init__(self):
        super(PointerAttention, self).__init__()
        # attention的累积分布加入attention weight的计算，防止重复生成相同的词或短语。同时在计算损失时，加入coverage loss。
        if config.do_coverage:
            self.W_c = nn.Linear(1, config.d_model, bias=False)
        # 线性变化
        self.decode_proj = nn.Linear(config.d_model, config.d_model)
        self.v = nn.Linear(config.d_model, 1, bias=False)

    def forward(self, s_t, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        """decoder运行到t时刻时，计算此时的attention weight。

        :param s_t: 来自decoder，输入计算attention weight。维度：B x d_model
        :param encoder_outputs: 计算context vector。B x seq_len x d_model
        :param encoder_feature: Wh * hi 进行attention weight计算的部分
        :param enc_padding_mask: 记录输入文本的padding的mask。
        :param coverage: 前t-1步累计attention dist
        :return:
            context_v, context vector
            attn_dist, decoder运行到t时刻的attention分布
            coverage, attention的累积分布
        """
        # seq_len: 需要summary的文本的tokens的长度。文章中，encoder中i为e_i维度上的index
        b, seq_len, n = list(encoder_outputs.size())  # n = d_model

        # decoder输出，计算attention weight的数据
        dec_fea = self.decode_proj(s_t)                                    # B x d_model
        dec_fea_expanded = dec_fea.unsqueeze(1).expand(b, seq_len, n).contiguous() # B x seq_len x d_model
        dec_fea_expanded = dec_fea_expanded.view(-1, n)                        # B*seq_len x d_model
        att_features = encoder_feature + dec_fea_expanded                      # B*seq_len x d_model

        if config.do_coverage:
            coverage_input = coverage.view(-1, 1)                              # B*seq_len x 1
            coverage_feature = self.W_c(coverage_input)                        # B*seq_len x d_model
            att_features = att_features + coverage_feature

        # attention weight
        e = F.tanh(att_features)                                               # B * seq_len x d_model
        scores = self.v(e)                                                     # B * seq_len x 1
        scores = scores.view(-1, seq_len)                                          # B x seq_len

        # padding部分处理
        attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask               # B x seq_len
        normalization_factor = attn_dist_.sum(1, keepdim=True)
        attn_dist = attn_dist_ / normalization_factor                          # B x seq_len

        # 计算context vector
        attn_dist = attn_dist.unsqueeze(1)                                     # B x 1 x seq_len
        # batch matrix-matrix，does not broadcast.
        # [B x 1 x seq_len] bmm [B x seq_len x d_model]
        context_v = torch.bmm(attn_dist, encoder_outputs)                      # B x 1 x d_model
        context_v = context_v.view(-1, config.d_model)                         # B x d_model

        attn_dist = attn_dist.view(-1, seq_len)                                # B x seq_len

        # 更新coverage
        if config.do_coverage:
            coverage = coverage.view(-1, seq_len)
            coverage = coverage + attn_dist

        return context_v, attn_dist, coverage


class PointerDecoder(nn.Module):
    def __init__(self):
        super(PointerDecoder, self).__init__()
        self.attention_network = PointerAttention()
        self.x_context = nn.Linear(config.d_model + config.emb_dim, config.emb_dim)
        self.decoder = Decoder(config.num_layers, config.n_head, config.d_model,
                               config.vocab_size, config.max_dec_len,
                               d_q, d_k, d_v, config.d_affine,
                               config.embedding_dropout, config.att_dropout, config.fc_dorpout)

        # 输入：context vector，decoder的h和c，以及t时刻的decoder的输入x，计算p_gen。
        if config.pointer_gen:
            self.p_gen_linear = nn.Linear(config.d_model * 2 + config.emb_dim, 1)

        # 每一步计算词表上的预测分布
        self.out1 = nn.Linear(config.d_model * 2, config.d_model)
        nn.init.normal_(self.out1.weight)
        nn.init.zeros_(self.out1.bias)
        self.out2 = nn.Linear(config.d_model, config.vocab_size)
        nn.init.normal_(self.out2.weight)
        nn.init.zeros_(self.out2.bias)
        self.dropout = nn.Dropout(0.2)

    def forward(self, d_inp, d_len, s_t, enc_output, enc_feature, look_ahead_mask, enc_mask, extra_zeros,
                context_v, enc_batch_extend_vocab, coverage, step):
        if not self.training and step == 0:
            context_v_t, _, coverage_next = self.attention_network(s_t, enc_output, enc_feature,
                                                                enc_mask, coverage)
            coverage = coverage_next

        # 计算 attention 部分
        x_emb_t = self.decoder.embedding.word_embedding[d_len]
        x = self.x_context(torch.cat((context_v, x_emb_t), 1))
        s_t = self.decoder(d_inp[:, :d_len+1], enc_output, look_ahead_mask, enc_mask)  # (batch_size, seq_len, d_model)
        s_t = s_t[:, -1, :]
        context_v_t, attn_dist, coverage_next = self.attention_network(s_t, enc_output, enc_feature,
                                                                    enc_mask, coverage)
        if self.training or step > 0:
            coverage = coverage_next

        # 计算使用生成的word的概率。
        p_gen = None
        if config.pointer_gen:
            p_gen_input = torch.cat((context_v_t, s_t, x), 1)  # B x (2 * d_model + emb_dim)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = F.sigmoid(p_gen)

        # 计算generator的输出词分布
        output = torch.cat((s_t, context_v_t), 1)  # B x (d_model * 2)
        output = self.out1(output)                     # B x d_model
        output = self.dropout(output)
        output = self.out2(output)                     # B x vocab_size
        vocab_dist = F.softmax(output, dim=1)

        # 计算加权的最终分布
        if config.pointer_gen:
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist
            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)              # B x (vocab_size+extra_zeros num)
            # scatter_add: 按照enc_batch_extend_vocab给出的index，将attention distribution value加到vocab distribution中。
            final_dist = vocab_dist_.scatter_add(1, enc_batch_extend_vocab, attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, s_t, context_v_t, attn_dist, p_gen, coverage


class Model:
    def __init__(self, model_file_path=None, is_eval=False):
        self.encoder = PointerEncoder()
        self.decoder = PointerDecoder()
        self.encoder.embedding.weight = self.decoder.embedding.weight

        if use_cuda:
            self.encoder = self.encoder.cuda()
            self.decoder = self.decoder.cuda()

        if is_eval:
            self.encoder = self.encoder.eval()
            self.decoder = self.decoder.eval()

        if model_file_path is not None:
            state = torch.load(model_file_path, map_location=lambda storage, location: storage)
            self.encoder.load_state_dict(state['encoder_state_dict'])
            self.decoder.load_state_dict(state['decoder_state_dict'], strict=False)