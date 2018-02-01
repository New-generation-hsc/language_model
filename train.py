from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
from model import model
from utils import process
import argparse


parse = argparse.ArgumentParser()
parse.add_argument("--data_dir", type=str, default='data',
                   help="data directory. should contain train.txt/valid.txt/test.txt")
parse.add_argument("--rnn_size", type=int, default=650, help="size of LSTM internal state")
parse.add_argument("--char_embed_size", type=int, default=15, help="dimensionality of character embedding")
parse.add_argument("--highway_layers", type=int, default=2, help="number of highway layers")
parse.add_argument("--kernels", type=str, default='[1, 2, 3, 4, 5, 7]', help="CNN kernel widths")
parse.add_argument("--kernel_features", type=str, default='[50, 100, 150, 200, 200, 200, 200]',
                   help="number of features in the CNN kernels")
parse.add_argument("--learning_rate_decay", type=float, default=0.5, help="learning rate decay")
parse.add_argument("--learning_rate", type=float, default=1.0, help="starting learning rate")
parse.add_argument("--decay_when", type=float, default=1.0,
                   help="decay if validation perplexity does not improve by more than this much")
parse.add_argument("--param_init", type=float, default=0.05, help="initialize parameters at")
parse.add_argument("--num_unroll_steps", type=int, default=35, help="number of time steps to unroll for")
parse.add_argument("--batch_size", type=int, default=20, help="number of sequences to train on in parallel")
parse.add_argument("--max_epochs", type=int, default=25, help="number of full passes")
parse.add_argument("--max_grad_norm", type=float, default=5.0, help="normalize gradients at")
parse.add_argument("--max_word_length", type=int, default=65, help="maximum word length")
parse.add_argument('--seed', type=int, default=3435, help='random number generator seed')
parse.add_argument('--print_every', type=str, default=5, help='how often to print current loss')
parse.add_argument('--EOS', type=str, default='+',
                   help='<EOS> symbol. should be a single unused character (like +) for PTB and blank for others')

args = parse.parse_args()


def str2list(x):
    """Convert the argument str into list"""
    x = x.strip()[1:-1]
    out = [int(i) for i in x.split(', ')]
    return out


transform = process.DataTransform(args.max_word_length)
reader = process.DataReader("data", ["train", "valid", "test"], transform=transform)

max_length, word_vocab, char_vocab, \
 word_tensor, char_tensor = reader.load()


model = model.Model(char_vocab.size, word_vocab.size)


def train():
    """
    every train in the data set
    :return:
    """
    average_loss = 0.
    pass