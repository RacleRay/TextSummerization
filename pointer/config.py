# -*- coding:utf-8 -*-
# author: Racle
# project: pointer-network

import os

root_dir = os.path.expanduser(r"./")

# train_data_path = os.path.join(root_dir, "ptr_nw/cnn-dailymail-master/finished_files/train.bin")
train_data_path = os.path.join(root_dir, r"trainset")
eval_data_path = os.path.join(root_dir, r"val")
decode_data_path = "./val"
vocab_path = r"./vocab/vocab_file"
log_root = r"./log"

# Hyperparameters
hidden_dim = 256  # LSTM hidden size
emb_dim = 128  # 输入词嵌入维度
batch_size = 8
max_enc_steps = 500
max_dec_steps = 60
beam_size = 4
min_dec_steps = 10
vocab_size = 5000

lr=0.15
adagrad_init_acc = 0.1
weight_decay=0.0001
norm_init_std = 1e-4  # 输入词嵌入初始化方差，可以选则使用预训练词向量
max_grad_norm = 3.0

pointer_gen = True  # 使用pointer generator结构
do_coverage = True  # 使用Coverage mechanism
cov_loss_wt = 1.0  # Coverage loss的权重
# 原文使用方法：先不使用Coverage mechanism，在合适轮次开启Coverage mechanism再训练。
# Note, the experiments reported in the ACL paper train WITHOUT coverage until converged,
# and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results
# in the ACL paper, turn this off for most of training then turn on for a short phase at the end.

eps = 1e-12  # 防止log运算中出现0
max_iterations = 80000

use_gpu = True

lr_coverage = 0.02  # 使用Coverage mechanism时的学习率
