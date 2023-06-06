import os
import sys
import time
import datetime
import argparse
import jsonlines

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import random
import numpy as np

from tqdm import trange

from model.superbert import SuperBertModel
from utils import *
from datasets.dataset import BertEvalDataset, BertTrainDataset
from datasets.negative_sampler import RandomSampler, PopularSampler

from options import args

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, list) else choice(list(x))

def str2bool(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

class SearcherEvolution():
    def __init__(self, args):
        self.args = args
        # information that evolutional algorithm needs
        self.select_num = args.select_num
        self.warmup_epochs = args.warmup_epochs
        self.search_epochs = args.search_epochs
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.num_layers = args.num_layers

        # generate dataset
        self.dataset = self._generate_dataset(args.dataset)
        [user_train, user_valid, user_test, usernum, itemnum] = self.dataset
        train_sampler = RandomSampler(user_train, user_valid, user_test, usernum, itemnum, args.train_negative_sample_size)
        eval_sampler = PopularSampler(user_train, user_valid, user_test, usernum, itemnum, args.eval_negative_sample_size)
        warp_dataset = BertTrainDataset(user_train, user_valid, user_test, usernum, itemnum, args.maxlen, train_sampler, args.mask_prob, args.dataset_random_seed, dupe_factor=args.dupe_factor, prop_sliding_window=args.prop_sliding_window)
        # dist_train_sampler = torch.utils.data.distributed.DistributedSampler(warp_dataset)
        # self.train_loader = DataLoader(warp_dataset, batch_size=args.batch_size, num_workers=4, sampler=dist_train_sampler)
        self.train_loader = DataLoader(warp_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)
        val_dataset = BertEvalDataset(user_train, user_valid, user_test, usernum, itemnum, args.maxlen, eval_sampler, mode='val', eval_set=args.eval_set_size)
        self.val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, num_workers=4, shuffle=False)
        test_dataset = BertEvalDataset(user_train, user_valid, user_test, usernum, itemnum, args.maxlen, eval_sampler, mode='test', eval_set=args.eval_set_size)
        self.test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=4, shuffle=False)

        # search space
        self.block_choice = [0, 1]
        self.rec_choice = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
        self.ind_choice = [0, 0.0001, 0.0005, 0.001, 0.0015, 0.002]
        self.rec_weights = [0 for _ in range(args.num_layers)]
        self.ind_weights = [0 for _ in range(args.num_layers)]
        self.lambda_choice = self.rec_choice + self.ind_choice
        
        # generate model
        self.model = self._generate_supernet(args)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

        # population info
        self.memory = []
        self.epoch = 0
        self.candidates = []
        self.keep_top_k = {self.select_num: []}
        self.vis_dict = {}
        
        # scale factor
        self.scale_factor = args.scale_factor
        self.scale_decay_rate = args.scale_decay_rate
        # self.hsic_learner = RbfHSIC()

    def _generate_supernet(self, args):
        [user_train, user_valid, user_test, usernum, itemnum] = self.dataset
        model = SuperBertModel(usernum, itemnum, self.rec_choice, self.ind_choice, args).to(args.device)
        # init model
        for name, param in model.named_parameters():
            try:
                if ('layer_norm' not in name) and ('bias' not in name):
                    torch.nn.init.trunc_normal_(param.data, std=args.initializer_range)
            except:
                pass # just ignore those failed init layers
        return model

    def _generate_dataset(self, dataset):
        dataset = data_partition(self.args.dataset)
        return dataset

    def _get_weight(self, choices, prob):
        '''
        prob is a value from [0, 1], means the weight for the auxiliary loss
        choices is actually a sampled point of the function that measures the loss weight and the prob
        after the prob is given, we get the final result through the sampled choices interpolation
        '''
        # prob = min(prob, 1-1e-10)   # Prevent bugs, but this step is not required in probability
        split_value = 1 / (len(choices) - 1)
        idx = 0
        while(prob > split_value):
            idx += 1
            prob -= split_value
        relate_distance = prob / split_value
        return choices[idx] * (1 - relate_distance) + choices[idx + 1] * relate_distance

    def _set_choice(self, cand):
        # change cand to weight
        num_layers = int(len(cand) / 2)
        block_cand = []
        for i in range(0, 2 * num_layers, 2):
            rec = cand[i]
            ind = cand[i + 1]
            rec_weight = self._get_weight(self.rec_choice, rec)
            ind_weight = self._get_weight(self.ind_choice, ind)
            self.rec_weights[int(i/2)] = rec_weight
            self.ind_weights[int(i/2)] = ind_weight
            block_cand.append(rec_weight)
            block_cand.append(ind_weight)
        # set block
        self.model.set_choice(np.array(block_cand))

    def sample_random(self):
        res = list()
        for _ in range(self.args.num_layers):
            res.append(random.random())
            res.append(random.random())
        return res

    def stack_random_cand(self, random_func, *, batch_size=10):
        while True:
            cands = [random_func() for _ in range(batch_size)]
            for cand in cands:
                if str(cand) not in self.vis_dict:
                    self.vis_dict[str(cand)] = {}
                info = self.vis_dict[str(cand)]
            for cand in cands:
                yield cand
    
    def get_cand_auc(self, cand):
        self._set_choice(cand)
        self.model.eval()
        t_valid, AUC = evaluate_loader(self.model, self.val_loader, self.args, 'val', ks=[10])
        self.vis_dict[str(cand)]['V_NDCG'] = float(t_valid[0][10])
        self.vis_dict[str(cand)]['V_HR'] = float(t_valid[1][10])
        self.vis_dict[str(cand)]['V_AUC'] = float(AUC)
        return float(AUC)

    def check_cand(self, cand):
        # assert isinstance(cand, list) and len(cand) == (self.num_layers * 2)
        if str(cand) not in self.vis_dict:
            self.vis_dict[str(cand)] = {}
        info = self.vis_dict[str(cand)]
        if 'visited' in info:
            return False
        info['visited'] = True
        info['auc'] = float(self.get_cand_auc(cand))
        return True
    
    def get_random(self, population_num):
        '''
        sample random candidates
        '''
        print('random select ......')
        cand_iter = self.stack_random_cand(self.sample_random)
        max_iter = (population_num - len(self.candidates) + 1) * 50
        while len(self.candidates) < population_num and max_iter > 0:
            max_iter -= 1
            cand = next(cand_iter)
            if not self.check_cand(cand):
                continue
            self.candidates.append(cand)
            print(f'random {len(self.candidates)} / {population_num}')
        print(f'sample over ...... random_num = {len(self.candidates)}')

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print(f'select top-{k} ...')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def show_top_k(self, k):
        assert k in self.keep_top_k
        t = self.keep_top_k[k]
        for cand in t:
            info = self.vis_dict[str(cand)]
            if 'test_auc' not in info:
                self._set_choice(cand)
                self.model.eval()
                t_test, AUC = evaluate_loader(self.model, self.test_loader, self.args, mode='test', ks=[10])
                info['T_NDCG'] = float(t_test[0][10])
                info['T_HR'] = float(t_test[1][10])
                info['T_AUC'] = float(AUC)
                info['test_auc'] = float(AUC)
    
    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        max_iter = crossover_num * 10
        def random_func():
            c1 = choice(self.keep_top_k[k])
            c2 = choice(self.keep_top_k[k])
            return list(choice([i,j]) for i, j in zip(c1, c2))
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iter > 0:
            max_iter -= 1
            cand = next(cand_iter)
            if not self.check_cand(cand):
                continue
            res.append(cand)
            print(f"crossover {len(res)} / {crossover_num}")
        print(f"crossover over ...... crossover_num = {len(res)}")
        return res
    
    def get_mutation(self, k, mutation_num, m_prob):
        '''
        mutation
        '''
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        max_iter = mutation_num * 10
        def random_func():
            # get top k candidates
            cand = list(choice(self.keep_top_k[k]))
            for i in range(self.num_layers * 2):
                if np.random.random_sample() < m_prob:
                    cand2 = list(choice(self.keep_top_k[k]))
                    cand3 = list(choice(self.keep_top_k[k]))
                    mutation_value = cand[i] + self.scale_factor * (cand2[i] - cand3[i])
                    cand[i] = min(1 - 1e-10, max(1e-10, mutation_value))
            return cand
        cand_iter = self.stack_random_cand(random_func)
        while len(res) < mutation_num and max_iter > 0:
            max_iter -= 1
            cand = next(cand_iter)
            if not self.check_cand(cand):
                continue
            res.append(cand)
            print(f'mutation {len(res)} / {mutation_num}')
        print(f"mutation over ...... mutation_num = {len(res)}")
        return res

    def _train_warmup(self):
        [user_train, user_valid, user_test, usernum, itemnum] = self.dataset
        # sampler = RandomSampler(user_train, usernum, itemnum, batch_size=self.args.batch_size, maxlen=self.args.maxlen, n_workers=3)
        num_batch = len(user_train) // self.args.batch_size
        cc = 0.0
        for u in user_train:
            cc += len(user_train[u])
        # f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
        T = 0.0
        t0 = time.time()
        for epoch in range(self.args.warmup_epochs):
            # sample a candidate
            self.model.train()
            cand = self.sample_random()
            self._set_choice(cand)
            with tqdm.tqdm(self.train_loader) as t:
                for batch, labels in t:
                    t.set_description(f"Warmup epoch {epoch + 1} / {self.args.warmup_epochs} ")
                    s = time.time()
                    seq, deq = batch
                    seq_pos_ids = torch.LongTensor(np.tile(np.array(range(seq.shape[1])), [seq.shape[0], 1])).to(self.args.device)
                    seq_sent_ids = torch.zeros(seq.shape, dtype=int).to(self.args.device)
                    deq_pos_ids = torch.LongTensor(np.tile(np.array(range(deq.shape[1])), [deq.shape[0], 1])).to(self.args.device)
                    deq_sent_ids = torch.zeros(seq.shape, dtype=int).to(self.args.device)
                    self.optimizer.zero_grad()
                    logits, encoder_layer_input, decoder_layer_output, rec_layer_hsic = self.model(seq, deq, seq_pos_ids, seq_sent_ids, deq_pos_ids, deq_sent_ids)
                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1).to(logits.device)
                    loss = self.loss(logits, labels)
                    if len(encoder_layer_input) != 0 and len(encoder_layer_input) == len(decoder_layer_output):
                        for i in range(len(encoder_layer_input)):
                            loss += self.rec_weights[i] * F.mse_loss(encoder_layer_input[i], decoder_layer_output[i])
                    if self.args.num_heads > 1:
                        batch_size = rec_layer_hsic[0].shape[0]
                        label = torch.arange(self.args.num_heads)
                        label = torch.tile(label, [batch_size * self.args.maxlen, 1]).to(self.args.device)
                        for l in range(len(rec_layer_hsic)):
                            loss += self.ind_weights[i] * F.nll_loss(rec_layer_hsic[l].view(batch_size * self.args.maxlen, self.args.num_heads, self.args.num_heads), label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
                    self.optimizer.step()
                    e = time.time()
                    t.set_postfix(loss=loss.item())
        print("over")

    def search(self):
        print(f"population_num = {self.population_num}, select_num = {self.select_num}, mutation_num = {self.mutation_num}, crossover_num = {self.crossover_num}")
        print(f"Now warmup supernet ... The warmup epoch is {self.warmup_epochs}")
        self._train_warmup()
        # save checkpoint
        os.makedirs('./checkpoint', exist_ok=True)
        torch.save(self.model.state_dict(), './checkpoint/super.pth')
        print(f"Now search the candidates ... The max epoch is {self.search_epochs}")
        # init population
        self.get_random(self.population_num)
        with trange(self.search_epochs) as t:
            for s_epoch in t:
                self.epoch += 1
                self.memory.append([])
                for cand in self.candidates:
                    self.memory[-1].append(cand)

                # update top k
                self.update_top_k(
                    self.candidates, k=self.select_num, key=lambda x:self.vis_dict[str(x)]['auc']
                )
                # now get mutation and crossover
                mutation = self.get_mutation(
                    self.select_num,
                    self.mutation_num,
                    self.m_prob
                )
                crossover = self.get_crossover(
                    self.select_num,
                    self.crossover_num
                )
                self.candidates = mutation + crossover
                self.get_random(self.population_num)
        # save top_k
        os.makedirs('./res', exist_ok=True)
        with jsonlines.open(f'./res/res_{self.args.dataset}_lr_{self.args.lr}_reg_{self.args.weight_decay}_warm_{self.args.warmup_epochs}_search_{self.args.search_epochs}_layers_{self.args.num_layers}_select_{self.args.select_num}_population_{self.args.population_num}_cross_{self.args.crossover_num}_mutation_{self.args.mutation_num}.jsonl', mode='w') as writer:
            t = self.keep_top_k[self.select_num]
            for cand in t:
                info = self.vis_dict[str(cand)]
                self._set_choice(cand)
                info['cand'] = str(cand)
                info['rec'] = str(self.rec_weights)
                info['ind'] = str(self.ind_weights)
                writer.write(info)

def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_rng_seed(args.seed)
    Searcher = SearcherEvolution(args)
    Searcher.search()

if __name__ == "__main__":
    main()