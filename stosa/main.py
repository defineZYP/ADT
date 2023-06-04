# -*- coding: utf-8 -*-

import os
import numpy as np
import random
import torch
import pickle
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import DisenDataset
from trainer import DistSAModelTrainer
from models import SASRecModel, DisenDistSAModel, DistMeanSAModel
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed, get_lambdas, set_template

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='./data/', type=str)
    parser.add_argument('--output_dir', default='./experiment/', type=str)
    parser.add_argument('--dataset', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", type=str, default='adt')
    parser.add_argument("--hidden_units", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_dropout", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--dropout", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    parser.add_argument('--pvn_weight', default=0.1, type=float)
    parser.add_argument('--kernel_param', default=1.0, type=float)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=256, help="number of batch_size")
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--eval_set", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=400, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    parser.add_argument('--topk', type=int, default=-1)

    args = parser.parse_args()
    args = set_template(args)
    print(args)

    set_seed(args.seed)
    check_path(args.output_dir)

    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + args.dataset + '.txt'
    #item2attribute_file = args.data_dir + args.dataset + '_item2attributes.json'

    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
        get_user_seqs(args.data_file)

    #item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1
    #args.attribute_size = attribute_size + 1

    # save model args
    args_str = f'{args.model_name}-{args.dataset}-{args.hidden_units}-{args.num_layers}-{args.num_heads}-{args.hidden_act}-{args.attention_dropout}-{args.dropout}-{args.maxlen}-{args.lr}-{args.weight_decay}-{args.ckp}-{args.kernel_param}-{args.pvn_weight}'
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    # print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    #args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = DisenDataset(args, user_seq, data_type='train')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    eval_dataset = DisenDataset(args, user_seq, data_type='valid', eval_set=args.eval_set)
    eval_sampler = SequentialSampler(eval_dataset)
    #eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=200)

    test_dataset = DisenDataset(args, user_seq, data_type='test', eval_set=args.eval_set)
    test_sampler = SequentialSampler(test_dataset)
    #test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=200)
    lambda1, lambda2 = get_lambdas(args.dataset, args.topk)

    model = DisenDistSAModel(args=args)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
    trainer = DistSAModelTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args, lambda1, lambda2)

    if args.do_eval:
        trainer.load(args.checkpoint_path)
        valid_scores, _, _ = trainer.valid('best', full_sort=True)
        # print(f'Load model from {args.checkpoint_path} for test!')
        scores, result_info, _ = trainer.test(0, full_sort=True)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=100, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # evaluate on MRR
            scores, _, _ = trainer.valid(epoch, full_sort=True)
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                # print("Early stopping")
                break

        # print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        valid_scores, _, _ = trainer.valid('best', full_sort=True)
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info, _ = trainer.test('best', full_sort=True)

    return valid_scores, scores

# hit1, ndcg1, hit5, ndcg5, hit10, ndcg10, hit15, ndcg15, hit20, ndcg20, hit40, ndcg40, mrr = main()
# print(hit1, ndcg1, hit5, ndcg5, hit10, ndcg10, hit15, ndcg15, hit20, ndcg20, hit40, ndcg40, mrr)
valid_scores, scores = main()
print(f'({valid_scores[0]}, {valid_scores[2]}, {valid_scores[3]}, {valid_scores[-1]}, {scores[0]}, {scores[2]}, {scores[3]}, {scores[-1]})')
