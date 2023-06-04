import os
import sys
import time
import datetime
import argparse
import jsonlines

import random
import numpy as np

from utils import *
from datasets import DisenDataset

from supernet import DisenDistSASupernet
from searcher import SearcherEvolution

def str2bool(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args():
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument("--output_dir", default="./experiment_ev/", type=str)
    parser.add_argument("--dataset", type=str, default="Beauty")
    parser.add_argument("--maxlen", type=int, default=100)
    parser.add_argument("--no_cuda", action="store_true")

    # trainer
    parser.add_argument("--pvn_weight", type=float, default=0.005)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="number of eval_batch_size")
    parser.add_argument("--eval_set", type=int, default=-1)
    # parser.add_argument("--epochs", type=int, default=500, help="number of epochs to train supernet")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")

    # model
    parser.add_argument("--hidden_units", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_layers", type=int, default=1, help="number of layers")
    parser.add_argument('--num_heads', default=4, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_dropout", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    parser.add_argument('--kernel_param', default=1.0, type=float)

    # evolution
    parser.add_argument('--warmup_epochs', default=200, type=int)
    parser.add_argument('--search_epochs', default=50, type=int)
    parser.add_argument('--population_num', type=int, default=20)
    parser.add_argument('--select_num', type=int, default=10)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover_num', type=int, default=5)
    parser.add_argument('--mutation_num', type=int, default=5)
    parser.add_argument('--scale_factor', type=float, default=0.5)
    parser.add_argument('--scale_decay_rate', type=float, default=0.5)

    args = parser.parse_args()

    set_seed(args.seed)
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    args.data_file = args.data_dir + args.dataset + '.txt'

    print(args)
    return args


if __name__ == "__main__":
    args = parse_args()
    searcher = SearcherEvolution(args)
    searcher.search()
