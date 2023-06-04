import sys
import copy
import json
import math
import time
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import random
import numpy as np
from tqdm import trange
from collections import defaultdict, Counter
from multiprocessing import Process, Queue

import matplotlib.pyplot as plt
lst = []

class PopularSampler:
    def __init__(self, train, val, test, usernum, itemnum, sample_size):
        self.train = train
        self.val = val
        self.test = test
        self.usernum = usernum
        self.itemnum = itemnum
        self.sample_size = sample_size
        self.popular_items, self.popular_p = self._generate_popular_items()
        self.negative_samples = self._generate_negative_samples()

    def _generate_popular_items(self):
        popularity = Counter()
        for user in range(1, self.usernum + 1):
            popularity.update(self.train[user])
            popularity.update(self.val[user])
            popularity.update(self.test[user])
        popular_items = sorted(popularity, key=popularity.get, reverse=True)
        popular_p = [popularity[i] for i in range(self.itemnum)]
        popular_p = popular_p / np.sum(popular_p)
        return popular_items, popular_p

    def _generate_negative_samples(self):
        # static sample, no use any more
        return None
    
    def _no_negatvie_sample(self, user, mode='valid'):
        item_idx = []
        if mode == 'val':
            seen = self.val[user]
        if mode == 'test':
            seen = self.test[user]
        for i in range(1, self.itemnum + 1):
            if i not in seen:
                item_idx.append(i)
        np.random.shuffle(item_idx)
        return item_idx[:self.itemnum-1]

    def get_negative_samples(self, user, mode="valid"):
        if self.sample_size < 0:
            return self._no_negatvie_sample(user, mode)
        item_idx = []
        seen = set(self.train[user])
        seen.update(self.val[user])
        if mode == 'test':
            seen.update(self.test[user])
        while len(item_idx) < self.sample_size:
            sampled_ids = np.random.choice(list(range(self.itemnum)), 2 * self.sample_size, replace=False, p=self.popular_p)
            sampled_ids = [x for x in sampled_ids if x not in seen and x not in item_idx]
            item_idx.extend(sampled_ids[:])
        return item_idx[:self.sample_size]

# sampler for batch generation
# negative sampler
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t

def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        deq = np.zeros([maxlen + 1], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        lst.append(user)
        for i in reversed(user_train[user][:-1]):
            deq[idx + 1] = i
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, deq[:-1], pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

class EvalDataset(Dataset):
    def __init__(self, user_train, user_val, user_test, usernum, itemnum, maxlen, negative_sampler, mode='val', eval_set=None):
        self.user_train = user_train
        self.user_val = user_val
        self.user_test = user_test
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.negative_sampler = negative_sampler
        self.mode = mode
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
        label = [1]
        if self.mode == 'val':
            # popularity sample valid set
            seq = np.zeros([self.maxlen], dtype=np.int32)
            idx = self.maxlen - 1
            for i in reversed(self.user_train[user]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break
            item_idx = [self.user_val[user][0]]
            item_idx += self.negative_sampler.get_negative_samples(user, mode=self.mode)
        elif self.mode == 'test':
            # popularity sample test set
            seq = np.zeros([self.maxlen], dtype=np.int32)
            idx = self.maxlen - 1
            seq[idx] = self.user_val[user][0]
            idx -= 1
            for i in reversed(self.user_train[user]):
                seq[idx] = i
                idx -= 1
                if idx == -1:
                    break
            item_idx = [self.user_test[user][0]]
            item_idx += self.negative_sampler.get_negative_samples(user, mode=self.mode)
        item_idx = np.array(item_idx)
        label += [0 for _ in range(len(item_idx) - 1)]
        label = np.array(label)
        return (user, seq, item_idx, label)

    def __getitem__(self, i):
        user = self.users[i]
        # if self.mode == 'test':
        #     while len(self.user_train[user]) < 1 or len(self.user_test[user]) < 1:
        #         user = (user + 1) % self.usernum + 1
        # elif self.mode == 'val':
        #     while len(self.user_train[user]) < 1 or len(self.user_val[user]) < 1:
        #         user = (user + 1) % self.usernum + 1
        user, seq, item_idx, label = self.sample_data(user)
        return (user, seq, item_idx), label
    
    def __len__(self):
        return len(self.users)

class EvalDoubleDataset(Dataset):
    def __init__(self, user_train, user_val, user_test, usernum, itemnum, maxlen, negative_sampler, mode='val', eval_set=None):
        self.user_train = user_train
        self.user_val = user_val
        self.user_test = user_test
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        self.negative_sampler = negative_sampler
        self.mode = mode
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
        label = [1]
        if self.mode == 'val':
            # popularity sample valid set
            seq = np.zeros([self.maxlen], dtype=np.int32)
            deq = np.zeros([self.maxlen + 1], dtype=np.int32)
            idx = self.maxlen - 1
            for i in reversed(self.user_train[user]):
                seq[idx] = i
                deq[idx + 1] = i
                idx -= 1
                if idx == -1:
                    break
            item_idx = [self.user_val[user][0]]
            item_idx += self.negative_sampler.get_negative_samples(user, mode=self.mode)
        elif self.mode == 'test':
            # popularity sample test set
            seq = np.zeros([self.maxlen], dtype=np.int32)
            deq = np.zeros([self.maxlen + 1], dtype=np.int32)
            idx = self.maxlen - 1
            seq[idx] = self.user_val[user][0]
            deq[idx + 1] = self.user_val[user][0]
            idx -= 1
            for i in reversed(self.user_train[user]):
                seq[idx] = i
                deq[idx + 1] = i
                idx -= 1
                if idx == -1:
                    break
            item_idx = [self.user_test[user][0]]
            item_idx += self.negative_sampler.get_negative_samples(user, mode=self.mode)
        item_idx = np.array(item_idx)
        label += [0 for _ in range(len(item_idx) - 1)]
        label = np.array(label)
        return (user, seq, deq[:-1], item_idx, label)

    def __getitem__(self, i):
        user = self.users[i]
        # if self.mode == 'test':
        #     while len(self.user_train[user]) < 1 or len(self.user_test[user]) < 1:
        #         user = (user + 1) % self.usernum + 1
        # elif self.mode == 'val':
        #     while len(self.user_train[user]) < 1 or len(self.user_val[user]) < 1:
        #         user = (user + 1) % self.usernum + 1
        user, seq, deq, item_idx, label = self.sample_data(user)
        return (user, seq, deq, item_idx), label
    
    def __len__(self):
        return len(self.users)

class WarpDataset(Dataset):
    def __init__(self, user_train, usernum, itemnum, maxlen):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen

    def sample_data(self, user):
        seq = np.zeros([self.maxlen], dtype=np.int32)
        deq = np.zeros([self.maxlen + 1], dtype=np.int32)
        pos = np.zeros([self.maxlen], dtype=np.int32)
        neg = np.zeros([self.maxlen], dtype=np.int32)
        nxt = self.user_train[user][-1]
        idx = self.maxlen - 1

        ts = set(self.user_train[user])
        lst.append(user)
        for i in reversed(self.user_train[user][:-1]):
            deq[idx + 1] = i
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, deq[:-1], pos, neg)
    
    def __getitem__(self, i):
        user = i % self.usernum + 1
        while len(self.user_train[user]) < 1: 
            user = np.random.randint(1, usernum + 1)
        data = self.sample_data(user)
        return data, 0

    def __len__(self):
        return self.usernum

# train/val/test data generation
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
    # return [user_train, user_valid, user_test, usernum, itemnum], User
    return user_train, user_valid, user_test, usernum, itemnum

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, negative_sampler, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    # if usernum>10000:
    #     users = random.sample(range(1, usernum + 1), 10000)
    # else:
    #     users = range(1, usernum + 1)
    users = range(1, usernum + 1)
    with trange(len(users)) as tra:
        tra.set_description("Evaluate: ")
        for step in tra:
            s = time.time()
            u = users[step]
            if len(train[u]) < 1 or len(test[u]) < 1: continue
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            seq[idx] = valid[u][0]
            idx -= 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break
            item_idx = [test[u][0]]
            item_idx += negative_sampler.get_negative_samples(u, mode='test')
            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0] # - for 1st argsort DESC

            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

    return NDCG / valid_user, HT / valid_user

def evaluate_loader(model, loader, args, mode='val', ks=[5, 10]):
    NDCG = {k: 0.0 for k in ks}
    HT = {k: 0.0 for k in ks}
    valid_user = 0.0
    ranks_all = []
    with torch.no_grad():
        with tqdm.tqdm(loader) as t:
            t.set_description(f"Evaluate {mode}: ")
            for batch, label in t:
                u, seq, item_idx = batch
                u, seq, item_idx = np.array(u), np.array(seq), np.array(item_idx)
                # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
                predictions = -model.predict(u, seq, item_idx)
                # for given metric, calculate it 
                # batch_size * 1
                rank = predictions.argsort(dim=1).argsort(dim=1)[:, 0]
                ranks_all.extend(rank.cpu().detach().numpy())
                valid_user += u.shape[0]
                for k in ks:
                    rank_matrix = torch.where(rank < k)
                    hit_matrix = rank[rank_matrix]
                    ndcg_matrix = 1 / torch.log2(hit_matrix + 2)
                    HT[k] += hit_matrix.shape[0]
                    NDCG[k] += ndcg_matrix.sum()
                # candidates_size = item_idx.shape[1]
                # AUC = 
    for k in ks:
        HT[k] = HT[k] / valid_user
        NDCG[k] = NDCG[k].item() / valid_user
    ranks_all = np.array(ranks_all) + 1
    MRR = (1 / ranks_all).mean()
    candidates_size = 1 + item_idx.shape[1]
    AUC = np.mean((candidates_size - ranks_all) / (candidates_size - 1))
    return (NDCG, HT), AUC

def evaluate_loader_db(model, loader, args, mode='val', ks=[5, 10]):
    NDCG = {k: 0.0 for k in ks}
    HT = {k: 0.0 for k in ks}
    valid_user = 0.0
    ranks_all = []
    with torch.no_grad():
        with tqdm.tqdm(loader) as t:
            t.set_description(f"Evaluate {mode}: ")
            for batch, label in t:
                u, seq, deq, item_idx = batch
                u, seq, deq, item_idx = np.array(u), np.array(seq), np.array(deq), np.array(item_idx)
                # predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
                predictions = -model.predict(u, seq, deq, item_idx)
                # for given metric, calculate it 
                # batch_size * 1
                rank = predictions.argsort(dim=1).argsort(dim=1)[:, 0]
                ranks_all.extend(rank.cpu().detach().numpy())
                valid_user += u.shape[0]
                for k in ks:
                    rank_matrix = torch.where(rank < k)
                    hit_matrix = rank[rank_matrix]
                    ndcg_matrix = 1 / torch.log2(hit_matrix + 2)
                    HT[k] += hit_matrix.shape[0]
                    NDCG[k] += ndcg_matrix.sum()
                # candidates_size = item_idx.shape[1]
                # AUC = 
    for k in ks:
        HT[k] = HT[k] / valid_user
        NDCG[k] = NDCG[k].item() / valid_user
    ranks_all = np.array(ranks_all) + 1
    MRR = (1 / ranks_all).mean()
    candidates_size = 1 + item_idx.shape[1]
    AUC = np.mean((candidates_size - ranks_all) / (candidates_size - 1))
    return (NDCG, HT), MRR

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users

def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    recall_dict = {}
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            #sum_recall += len(act_set & pred_set) / float(len(act_set))
            one_user_recall = len(act_set & pred_set) / float(len(act_set))
            recall_dict[i] = one_user_recall
            sum_recall += one_user_recall
            true_users += 1
    return sum_recall / true_users, recall_dict

def cal_mrr(actual, predicted):
    sum_mrr = 0.
    true_users = 0
    num_users = len(predicted)
    mrr_dict = {}
    for i in range(num_users):
        r = []
        act_set = set(actual[i])
        pred_list = predicted[i]
        for item in pred_list:
            if item in act_set:
                r.append(1)
            else:
                r.append(0)
        r = np.array(r)
        if np.sum(r) > 0:
            #sum_mrr += np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)[0]
            one_user_mrr = np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)[0]
            sum_mrr += one_user_mrr
            true_users += 1
            mrr_dict[i] = one_user_mrr
        else:
            mrr_dict[i] = 0.
    return sum_mrr / len(predicted), mrr_dict


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def ndcg_k(actual, predicted, topk):
    res = 0
    ndcg_dict = {}
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
        ndcg_dict[user_id] = dcg_k / idcg
    return res / float(len(actual)), ndcg_dict


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def get_full_sort_score(epoch, answers, pred_list, ks, _print=True):
    recall, ndcg, mrr = [], [], 0
    recall_dict_list = []
    ndcg_dict_list = []
    for k in [5, 10]:
        recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
        recall.append(recall_result)
        recall_dict_list.append(recall_dict_k)
        ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
        ndcg.append(ndcg_result)
        ndcg_dict_list.append(ndcg_dict_k)
    mrr, mrr_dict = cal_mrr(answers, pred_list)
    post_fix = {
        "Epoch": epoch,
        "HIT@5": '{:.8f}'.format(recall[0]), "NDCG@5": '{:.8f}'.format(ndcg[0]),
        "HIT@10": '{:.8f}'.format(recall[1]), "NDCG@10": '{:.8f}'.format(ndcg[1]),
        "MRR": '{:.8f}'.format(mrr)
    }
    # if _print:
    #     print(post_fix, flush=True)
    #     with open(self.args.log_file, 'a') as f:
    #         f.write(str(post_fix) + '\n')
    return [recall[0], ndcg[0], recall[1], ndcg[1], mrr], str(post_fix), [recall_dict_list, ndcg_dict_list, mrr_dict]

def evaluate_loader_full(epoch, model, loader, args, mode='val', ks=[5, 10]):
    pred_list = None
    answer_list = None
    with torch.no_grad():
        i = 0
        for batch in loader:
            # batch = tuple(t.to('cuda') for t in batch)
            u, seq, dec, pos, neg, answers = batch
            u, seq, answers = np.array(u), np.array(seq), np.array(answers)
            rank = -model.predict(u, seq, answers, True)
            rank = rank.cpu().data.numpy().copy()
            # print(rank.shape)
            # rank = predictions.argsort(dim=1).argsort(dim=1)[:, 0]
            # rank = rank.cpu().data.numpy().copy()
            batch_user_index = u
            rank[args.train_matrix[batch_user_index].toarray() > 0] = 1e+24
            ind = np.argpartition(rank, 40)[:, :40]
            arr_ind = rank[np.arange(len(rank))[:, None], ind]
            # ascending order
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rank)), ::]
            #arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rank)), ::-1]
            batch_pred_list = ind[np.arange(len(rank))[:, None], arr_ind_argsort]

            if i == 0:
                pred_list = batch_pred_list
                answer_list = answers
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                answer_list = np.append(answer_list, answers, axis=0)
            i += 1
        return get_full_sort_score(epoch, answer_list, pred_list, ks)

# evaluate on val set
def evaluate_valid(model, dataset, negative_sampler, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    users = range(1, usernum + 1)

    # if usernum>10000:
    #     users = random.sample(range(1, usernum + 1), 10000)
    # else:
    #     users = range(1, usernum + 1)

    with trange(len(users)) as tra:
        tra.set_description("Evaluate Valid: ")
        for step in tra:
            u = users[step]
            if len(train[u]) < 1 or len(valid[u]) < 1: continue

            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            for i in reversed(train[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            item_idx = [valid[u][0]]
            item_idx += negative_sampler.get_negative_samples(u, mode='valid')

            predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
            predictions = predictions[0]
            rank = predictions.argsort().argsort()[0].item()

            valid_user += 1

            if rank < 10:
                NDCG += 1 / np.log2(rank + 2)
                HT += 1

    return NDCG / valid_user, HT / valid_user

def pairwise_distances(x):
    #x should be two dimensional
    instances_norm = torch.sum(x**2,-1).reshape((-1,1))
    return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ /sigma)

def HSIC_loss(x, y, s_x=1, s_y=1):
    m,_ = x.shape #batch size
    K = GaussianKernelMatrix(x,s_x)
    L = GaussianKernelMatrix(y,s_y)
    H = torch.eye(m) - 1.0/m * torch.ones((m,m))
    H = H.float().cuda()
    HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)
    return HSIC

def distance_correlation_loss(x, y):
    def _create_centered_distance(X):
        r = torch.sum(torch.square(X), 1, keepdims=True)
        D = torch.sqrt(torch.maximum(r  - 2 * torch.matmul(X, X.t()) + r.t(), torch.tensor(0.0)) + 1e-8)

        D = D - torch.mean(D, dim=0, keepdims=True) - torch.mean(D, dim=1, keepdims=True) + torch.mean(D)
        return D
    
    def _create_distance_covariance(D1, D2):
        n_samples = D1.shape[0]
        dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2) / (n_samples * n_samples), torch.tensor(0.0)) + 1e-8)
        return dcov
    
    D1 = _create_centered_distance(x)
    D2 = _create_centered_distance(y)
    dcov_12 = _create_distance_covariance(D1, D2)
    dcov_11 = _create_distance_covariance(D1, D1)
    dcov_22 = _create_distance_covariance(D2, D2)
    dcor = dcov_12 / (torch.sqrt(torch.maximum(dcov_11 * dcov_22, torch.tensor(0.0))) + 1e-10)
    return dcor

def generate_filenames(args, base):
    file = base
    for key in vars(args):
        if key in ['lr', 'batch_size', 'maxlen', 'hidden_units', 'num_layers', 'dropout']:
            file += f'_{key}_{getattr(args, key)}'
    file += '.png'
    return file

def draw_figs(datas, title, save_path):
    plt.figure()
    rg = len(datas)
    x = list(range(rg))
    plt.plot(x, datas, 'g-', label='title')
    plt.legend()
    plt.xlabel('iters')
    plt.ylabel(title)
    plt.title(f"{title} per iter")
    plt.savefig(save_path)

def set_template(args, template_folder='./templates'):
    filename = f"{template_folder}/{args.dataset}.json"
    with open(filename, 'r') as f:
        arg_dic = json.load(f)
    for k in arg_dic:
        setattr(args, k, arg_dic[k])
    return args

def get_lambdas(dataset, tp=-1):
    '''
    按照dataset获取lambda_1(reconstruction)和lambda_2(independence)
    tp代表取top1、5或者average(-1)的lambda
    '''
    if dataset == 'ml-1m':
        return [0.104292, 0.065892], [0.100833, 0.000607]
    if dataset == 'beauty' or dataset == 'Beauty':
        return [0.0124, 0.122], [0.0001, 0.0]
    if dataset == 'steam':
        return [0.0001, 0.0005] , [0.00134, 0.00028]
    if dataset == 'ml-20m':
        return [0.005, 0.1], [0.00186667, 0.075]

if __name__ == "__main__":
    dataset = data_partition('ml-1m')
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    dataset = WarpDataset(user_train, usernum, itemnum, 200)
    train_loader = DataLoader(dataset=dataset, batch_size=256, shuffle=True, num_workers=2)
    size = 0
    for i in range(2):
        with tqdm.tqdm(train_loader) as t:
            for batch, label in t:
                u, seq, dec, pos, neg = batch
