import torch
import random

import numpy as np
import tqdm

from collections import defaultdict, Counter
from torch.utils.data import Dataset

from .negative_sampler import PopularSampler, RandomSampler

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    f = open('data/%s.txt' % fname, 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

class BertTrainDataset(Dataset):
    def __init__(self, user_train, user_val, user_test, usernum, itemnum, maxlen, negative_sampler, mask_prob, seed, generate=True, dupe_factor=10, prop_sliding_window=0.5):
        print(f"dataset: {usernum}~{itemnum}")
        self.user_train = user_train
        self.user_val = user_val
        self.user_test = user_test
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.negative_sampler = negative_sampler
        self.mask_prob = mask_prob
        self.rng = random.Random(seed)
        self.mask_token = self.itemnum + 1
        self.datas = []
        self.labels = []
        self.dupe_factor = dupe_factor
        self.prop_sliding_window = prop_sliding_window
        if generate:
            self._generate_data()
            self.dupe_factor = dupe_factor

    def _generate_data(self):
        '''
        generate data
        '''
        for user in tqdm.trange(self.usernum):
            # for each user, mask part of the sequence to get the training data
            seqs = self.user_train[user + 1]
            if len(seqs) < 1:
                continue
            # split to multi seq with maxlen
            if len(seqs) <= self.maxlen:
                for _ in range(self.dupe_factor):
                    data, label = self.sample_data(seqs)
                    self.datas.append(data)
                    self.labels.append(label)
            else:
                sliding_step = (int)(
                    self.prop_sliding_window * self.maxlen
                ) if self.prop_sliding_window != -1 else self.maxlen
                beg_idx = list(range(len(seqs) - self.maxlen, 0, -sliding_step))
                beg_idx.append(0)
                for i in beg_idx[::-1]:
                    seq = seqs[i: i + self.maxlen]
                    for _ in range(self.dupe_factor):
                        data, label = self.sample_data(seq)
                        self.datas.append(data)
                        self.labels.append(label)
            data, label = self._mask_last(seqs)
            self.datas.append(data)
            self.labels.append(label)

    def _mask_last(self, seq):
        '''
        according to bert4rec paper, there is a inconsistency between training and evaluation
        so we need to make training data which only mask the last item of the sequence .............. 
        '''
        tokens = []
        dec_tokens = []
        labels = []
        for s in seq:
            tokens.append(s)
            dec_tokens.append(s)
            labels.append(0)
        labels[-1] = seq[-1]
        tokens[-1] = self.mask_token
        dec_tokens[-1] = self.mask_token
        tokens = tokens[-self.maxlen:]
        labels = labels[-self.maxlen:]
        dec_tokens = dec_tokens[-self.maxlen:]
        mask_len = self.maxlen - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        dec_tokens = [0] * (self.maxlen - len(dec_tokens)) + dec_tokens
        return (torch.LongTensor(tokens), torch.LongTensor(dec_tokens)), torch.LongTensor(labels)

    def sample_data(self, seq):
        # popularity sample test set
        # from bert4rec
        tokens = []
        dec_tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                    dec_tokens.append(self.mask_token)
                elif prob < 0.9:
                    token = self.rng.randint(1, self.itemnum)
                    tokens.append(token)
                    dec_tokens.append(token)
                else:
                    tokens.append(s)
                    dec_tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                dec_tokens.append(s)
                labels.append(0)
        tokens = tokens[-self.maxlen:]
        labels = labels[-self.maxlen:]
        dec_tokens[-1] = self.mask_token
        dec_tokens = dec_tokens[-self.maxlen:]
        mask_len = self.maxlen - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        dec_tokens = [0] * (self.maxlen - len(dec_tokens)) + dec_tokens
        return (torch.LongTensor(tokens), torch.LongTensor(dec_tokens)), torch.LongTensor(labels)

    def __getitem__(self, i):
        seq, deq = self.datas[i]
        label = self.labels[i]
        return (seq, deq), label
    
    def __len__(self):
        return len(self.datas)
        # return self.datas.shape[0]
    
    def save_dataset(self, prefix):
        np.save(f"{prefix}_datas.npy", self.datas)
        np.save(f"{prefix}_labels.npy", self.labels)

    def load_dataset(self, prefix):
        self.datas = np.load(f"{prefix}_datas.npy", allow_pickle=True)
        self.labels = np.load(f"{prefix}_labels.npy", allow_pickle=True)

class BertEvalDataset(Dataset):
    def __init__(self, user_train, user_val, user_test, usernum, itemnum, maxlen, negative_sampler, mode='val', eval_set=-1):
        self.user_train = user_train
        self.user_val = user_val
        self.user_test = user_test
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.negative_sampler = negative_sampler
        self.mode = mode
        self.mask_token = self.itemnum + 1
        self.eval_set = self.usernum if eval_set is None else eval_set
        self.users = []
        if eval_set >= 0:
            tmp_users = random.sample(range(1, usernum + 1), eval_set)
        else:
            tmp_users = range(1, usernum + 1)
        for user in tmp_users:
            if mode == 'val':
                if len(self.user_val[user]) != 0 and len(self.user_train[user]) != 0:
                    self.users.append(user)
            elif mode == 'test':
                if len(self.user_test[user]) != 0 and len(self.user_train[user]) != 0:
                    self.users.append(user)

    def sample_data(self, user):
        # popularity sample test set
        seq = self.user_train[user]
        if self.mode == 'val':
            answer = [self.user_val[user][0]]
        else:
            answer = [self.user_test[user][0]]
        negs = self.negative_sampler.get_negative_samples(user, mode=self.mode)
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        seq = seq + [self.mask_token]
        seq = seq[-self.maxlen:]
        mask_len = self.maxlen - len(seq)
        seq = [0] * mask_len + seq
        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)
        
    def __getitem__(self, i):
        user = self.users[i]
        seq, candidates, labels = self.sample_data(user)
        return (user, seq, candidates), labels

    def __len__(self):
        return len(self.users)

class BertTuneDataset(Dataset):
    def __init__(self, user_train, user_val, user_test, usernum, itemnum, maxlen, negative_sampler, mask_prob, seed, generate=True):
        print(f"dataset: {usernum}~{itemnum}")
        self.user_train = user_train
        self.user_val = user_val
        self.user_test = user_test
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.negative_sampler = negative_sampler
        self.mask_prob = mask_prob
        self.rng = random.Random(seed)
        self.mask_token = self.itemnum + 1
        self.datas = []
        self.labels = []
        if generate:
            self._generate_data()

    def _generate_data(self):
        '''
        generate data
        '''
        for user in tqdm.trange(self.usernum):
            seqs = self.user_train[user + 1]
            if len(seqs) < 1:
                continue
            # split to multi seq with maxlen
            data, label = self._mask_last(seqs)
            self.datas.append(data)
            self.labels.append(label)

    def _mask_last(self, seq):
        tokens = []
        dec_tokens = []
        labels = []
        for s in seq:
            tokens.append(s)
            dec_tokens.append(s)
            labels.append(0)
        labels[-1] = seq[-1]
        tokens[-1] = self.mask_token
        dec_tokens[-1] = self.mask_token
        tokens = tokens[-self.maxlen:]
        labels = labels[-self.maxlen:]
        dec_tokens = dec_tokens[-self.maxlen:]
        mask_len = self.maxlen - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        dec_tokens = [0] * (self.maxlen - len(dec_tokens)) + dec_tokens
        return (torch.LongTensor(tokens), torch.LongTensor(dec_tokens)), torch.LongTensor(labels)

    def sample_data(self, seq):
        # popularity sample test set
        # seq = self.user_train[user]
        tokens = []
        dec_tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.mask_token)
                    dec_tokens.append(self.mask_token)
                elif prob < 0.9:
                    token = self.rng.randint(1, self.itemnum)
                    tokens.append(token)
                    dec_tokens.append(token)
                else:
                    tokens.append(s)
                    dec_tokens.append(s)
                labels.append(s)
            else:
                tokens.append(s)
                dec_tokens.append(s)
                labels.append(0)
        tokens = tokens[-self.maxlen:]
        labels = labels[-self.maxlen:]
        dec_tokens[-1] = self.mask_token
        dec_tokens = dec_tokens[-self.maxlen:]
        mask_len = self.maxlen - len(tokens)
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        dec_tokens = [0] * (self.maxlen - len(dec_tokens)) + dec_tokens
        return (torch.LongTensor(tokens), torch.LongTensor(dec_tokens)), torch.LongTensor(labels)

    def __getitem__(self, i):
        seq, deq = self.datas[i]
        label = self.labels[i]
        return (seq, deq), label
    
    def __len__(self):
        return len(self.datas)
        # return self.datas.shape[0]
    
    def save_dataset(self, prefix):
        np.save(f"{prefix}_datas.npy", self.datas)
        np.save(f"{prefix}_labels.npy", self.labels)

    def load_dataset(self, prefix):
        self.datas = np.load(f"{prefix}_datas.npy", allow_pickle=True)
        self.labels = np.load(f"{prefix}_labels.npy", allow_pickle=True)

class BertDoubleEvalDataset(Dataset):
    def __init__(self, user_train, user_val, user_test, usernum, itemnum, maxlen, negative_sampler, mode='val', eval_set=-1):
        self.user_train = user_train
        self.user_val = user_val
        self.user_test = user_test
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.negative_sampler = negative_sampler
        self.mode = mode
        self.mask_token = self.itemnum + 1
        self.eval_set = self.usernum if eval_set is None else eval_set
        self.users = []
        if eval_set >= 0:
            tmp_users = random.sample(range(1, usernum + 1), eval_set)
        else:
            tmp_users = range(1, usernum + 1)
        for user in tmp_users:
            if mode == 'val':
                if len(self.user_val[user]) != 0 and len(self.user_train[user]) != 0:
                    self.users.append(user)
            elif mode == 'test':
                if len(self.user_test[user]) != 0 and len(self.user_train[user]) != 0:
                    self.users.append(user)

    def sample_data(self, user):
        # popularity sample test set
        seq = self.user_train[user]
        deq = self.user_train[user]
        if self.mode == 'val':
            answer = [self.user_val[user][0]]
        else:
            answer = [self.user_test[user][0]]
        negs = self.negative_sampler.get_negative_samples(user, mode=self.mode)
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        seq = seq + [self.mask_token]
        deq[-1] = self.mask_token
        seq = seq[-self.maxlen:]
        deq = deq[-self.maxlen:]
        mask_len = self.maxlen - len(seq)
        seq = [0] * mask_len + seq
        mask_deq_len = self.maxlen - len(deq)
        deq = [0] * mask_deq_len + deq
        return torch.LongTensor(seq), torch.LongTensor(deq), torch.LongTensor(candidates), torch.LongTensor(labels)
        
    def __getitem__(self, i):
        user = self.users[i]
        seq, deq, candidates, labels = self.sample_data(user)
        return (user, seq, deq, candidates), labels

    def __len__(self):
        return len(self.users)