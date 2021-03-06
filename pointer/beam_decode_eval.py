# -*- coding:utf-8 -*-
# author: Racle
# project: pointer-network


import os
import time
import sys

import tensorflow as tf
import torch

import config
from data_loader import *
from model import Model
from utils import *

use_cuda = config.use_gpu and torch.cuda.is_available()


class Beam:
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens        # list
        self.log_probs = log_probs  # list
        self.state = state
        self.context = context
        self.coverage = coverage  # 前t-1步累计attention dist

    def extend(self, token, log_prob, state, context, coverage):
        "每一步计算下一个时刻beam size分支时，扩展候选分支"
        return Beam(tokens=self.tokens + [token],
                    log_probs=self.log_probs + [log_prob],
                    state=state,
                    context=context,
                    coverage=coverage)

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        self.model = Model(model_file_path, is_eval=True)

    def decode(self):
        """beam search并完成对结果的评价"""
        start = time.time()
        counter = 0
        batch = self.batcher.next_batch()
        while batch is not None:
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = outputids2words(output_ids, self.vocab, (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index(STOP_DECODING)
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            # list of list
            original_abstract_sents = batch.original_abstracts_sents[0]  # [0]: 取第一个batch
            # 保存解码结果和参考的summary
            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            print('===================')
            print('Ref:', original_abstract_sents)
            print('Predict:', ''.join(decoded_words))
            print('===================')
            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec' % (counter, time.time() - start))
                start = time.time()
            batch = self.batcher.next_batch()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)

    def beam_search(self, batch):
        "使用一份序列数据，复制beam_size份，进行解码"
        # batch should have only one example
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_0, coverage_t_0 = \
            get_encoder_variables(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        s_t_0 = self.model.reduce_state(encoder_hidden)

        dec_h, dec_c = s_t_0  # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context=c_t_0[0],
                      coverage=(coverage_t_0[0] if config.do_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t = y_t.cuda()

            # 迭代参数增加到beam_size
            all_state_h = []
            all_state_c = []
            all_context = []
            # beams：是复制了beam size个单输入解码结果的列表。
            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)
            s_t = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t = torch.stack(all_context, 0)
            coverage_t = None
            if config.do_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t = torch.stack(all_coverage, 0)

            final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t, s_t,
                                                                                    encoder_outputs, encoder_feature,
                                                                                    enc_padding_mask, c_t,
                                                                                    extra_zeros, enc_batch_extend_vocab,
                                                                                    coverage_t, steps)
            log_probs = torch.log(final_dist)   # B x (vocab_size+extra_zeros num)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)  # 返回value， index

            dec_h, dec_c = s_t
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            # 计算候选分支
            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):  # i: beam的id
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = c_t[i]
                coverage_i = (coverage_t[i] if config.do_coverage else None)
                for j in range(config.beam_size * 2):  # j：每个beam上候选的分支的id，（config.beam_size * 2个）
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                        log_prob=topk_log_probs[i, j].item(),
                                        state=state_i,
                                        context=context_i,
                                        coverage=coverage_i)
                    all_beams.append(new_beam)

            # 排序候选分支，并保留beam_size个分支
            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break
            # 完成一个解码步骤
            steps += 1

        if len(results) == 0:
            results = beams

        # 输出最佳结果
        beams_sorted = self.sort_beams(results)
        return beams_sorted[0]

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)


if __name__ == '__main__':
    # 运行: python model_filename
    # 只需要获得结果时，beam函数计算到best_summary = self.beam_search(batch)部分。
    model_filename = sys.argv[1]
    beam_Search_processor = BeamSearch(model_filename)
    beam_Search_processor.decode()