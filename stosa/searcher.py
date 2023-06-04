import numpy as np

import jsonlines
import random
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils import *
from datasets import DisenDataset

from supernet import DisenDistSASupernet
from super_trainer import SuperDistSAModelTrainer

choice = lambda x: x[np.random.randint(len(x))] if isinstance(
    x, list) else choice(list(x))
    
class SearcherEvolution():
    def __init__(self, args):
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
        user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
            get_user_seqs(args.data_file)
        args.item_size = max_item + 2
        args.num_users = num_users
        args.mask_id = max_item + 1
        args.train_matrix = valid_rating_matrix
        train_dataset = DisenDataset(args, user_seq, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = DisenDataset(args, user_seq, data_type='valid', eval_set=args.eval_set)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
        #eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=200)

        test_dataset = DisenDataset(args, user_seq, data_type='test', eval_set=args.eval_set)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
        # generate model
        self.rec_choice = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
        self.ind_choice = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
        self.rec_weights = [0 for _ in range(args.num_layers)]
        self.ind_weights = [0 for _ in range(args.num_layers)]
        self.lambda_choice = self.rec_choice + self.ind_choice
        model = DisenDistSASupernet(args, self.rec_choice, self.ind_choice)
        self.trainer = SuperDistSAModelTrainer(
            model, train_dataloader, eval_dataloader, test_dataloader, args
        )
        # population info
        self.memory = []
        self.epoch = 0
        self.candidates = []
        self.keep_top_k = {self.select_num: []}
        self.vis_dict = {}
        
        # scale factor
        self.scale_factor = args.scale_factor
        self.scale_decay_rate = args.scale_decay_rate
        self.args = args

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
        self.trainer.set_choice(np.array(block_cand))

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
    
    def get_cand_MRR(self, cand):
        self._set_choice(cand)
        scores, _, _ = self.trainer.valid(-1, full_sort=True)
        self.vis_dict[str(cand)]['V_NDCG'] = float(scores[5])
        self.vis_dict[str(cand)]['V_HR'] = float(scores[4])
        self.vis_dict[str(cand)]['V_MRR'] = float(scores[-1])
        return float(scores[-1])

    def check_cand(self, cand):
        # assert isinstance(cand, list) and len(cand) == (self.num_layers * 2)
        if str(cand) not in self.vis_dict:
            self.vis_dict[str(cand)] = {}
        info = self.vis_dict[str(cand)]
        if 'visited' in info:
            return False
        info['visited'] = True
        info['MRR'] = float(self.get_cand_MRR(cand))
        return True
    
    def get_random(self, population_num):
        '''
        sample random candidates
        '''
        # print('random select ......')
        cand_iter = self.stack_random_cand(self.sample_random)
        max_iter = (population_num - len(self.candidates) + 1) * 50
        while len(self.candidates) < population_num and max_iter > 0:
            max_iter -= 1
            cand = next(cand_iter)
            if not self.check_cand(cand):
                continue
            self.candidates.append(cand)
            # print(f'random {len(self.candidates)} / {population_num}')
        # print(f'sample over ...... random_num = {len(self.candidates)}')

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        # print(f'select top-{k} ...')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def show_top_k(self, k):
        assert k in self.keep_top_k
        t = self.keep_top_k[k]
        for cand in t:
            info = self.vis_dict[str(cand)]
            # self._set_choice(cand)
            if 'test_MRR' not in info:
                self._set_choice(cand)
                scores, _, _ = self.trainer.test(-1, full_sort=True)
                info['T_NDCG'] = float(scores[5])
                info['T_HR'] = float(scores[4])
                info['T_MRR'] = float(scores[-1])
                info['test_MRR'] = float(scores[-1])
    
    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        # print('crossover ......')
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
            # print(f"crossover {len(res)} / {crossover_num}")
        # print(f"crossover over ...... crossover_num = {len(res)}")
        return res
    
    def get_mutation(self, k, mutation_num, m_prob):
        '''
        mutation
        '''
        assert k in self.keep_top_k
        # print('mutation ......')
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
            # print(f'mutation {len(res)} / {mutation_num}')
        # print(f"mutation over ...... mutation_num = {len(res)}")
        return res

    def _train_warmup(self):
        with trange(self.args.warmup_epochs) as t:
            for epoch in t:
                cand = self.sample_random()
                self._set_choice(cand)
                self.trainer.train(epoch, lambda1=self.rec_weights, lambda2=self.ind_weights)

    def search(self):
        print(f"population_num = {self.population_num}, select_num = {self.select_num}, mutation_num = {self.mutation_num}, crossover_num = {self.crossover_num}")
        print(f"Now warmup supernet ... The warmup epoch is {self.warmup_epochs}")
        self._train_warmup()
        # torch.save(self.model.module.state_dict(), './checkpoint/super.pth')
        self.trainer.save_checkpoint('./checkpoint/super.pth')
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
                    self.candidates, k=self.select_num, key=lambda x:self.vis_dict[str(x)]['MRR']
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
        # self.show_top_k(self.select_num)
        # save top_k
        with jsonlines.open(f'./res/res_{self.args.data_name}_lr_{self.args.lr}_reg_{self.args.weight_decay}_warm_{self.args.warmup_epochs}_search_{self.args.search_epochs}_layers_{self.args.num_layers}_select_{self.args.select_num}_population_{self.args.population_num}_cross_{self.args.crossover_num}_mutation_{self.args.mutation_num}.jsonl', mode='w') as writer:
            t = self.keep_top_k[self.select_num]
            for cand in t:
                info = self.vis_dict[str(cand)]
                self._set_choice(cand)
                info['cand'] = str(cand)
                info['rec'] = str(self.rec_weights)
                info['ind'] = str(self.ind_weights)
                writer.write(info)

def _get_weight(choices, prob):
    # prob = min(prob, 1-1e-10)   # Prevent bugs, but this step is not required in probability
    split_value = 1 / (len(choices) - 1)
    idx = 0
    while(prob > split_value):
        idx += 1
        prob -= split_value
    relate_distance = prob / split_value
    return choices[idx] * (1 - relate_distance) + choices[idx + 1] * relate_distance

if __name__ == "__main__":
    rec_choice = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    ind_choice = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    cand = [0.7053411308078107, 0.9542592593410837, 0.9296478828883573, 0.28425047269448145, 0.1600125621449342, 0.47495464861462977]
    num_layers = int(len(cand) / 2)
    rec_weights = [0 for _ in range(num_layers)]
    ind_weights = [0 for _ in range(num_layers)]
    for i in range(0, 2 * num_layers, 2):
        rec = cand[i]
        ind = cand[i + 1]
        rec_weight = _get_weight(rec_choice, rec)
        ind_weight = _get_weight(ind_choice, ind)
        rec_weights[int(i/2)] = rec_weight
        ind_weights[int(i/2)] = ind_weight
    print(rec_weights, ind_weights)