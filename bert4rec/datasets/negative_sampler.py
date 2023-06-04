import numpy as np
from collections import defaultdict, Counter

class PopularSampler:
    '''
    sampling data according to the popularity
    '''
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
        '''
        calculate the number of the appearance of the item and treat it as a weight of possiblity
        '''
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

    def _no_negative_samples(self, user, mode='valid'):
        '''
        no sampling, treat all items as candidates
        '''
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

    def get_negative_samples(self, user, mode="test"):
        if self.sample_size < 0:
            return self._no_negative_samples(user, mode)
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

class RandomSampler:
    '''
    random sample data
    '''
    def __init__(self, train, val, test, usernum, itemnum, sample_size):
        self.train = train
        self.val = val
        self.test = test
        self.usernum = usernum
        self.itemnum = itemnum
        self.sample_size = sample_size
        self.negative_samples = self._generate_negative_samples()

    def _generate_negative_samples(self):
        # static sample, no use any more
        return None

    def get_negative_samples(self, user, mode="test"):
        item_idx = []
        seen = set(self.train[user])
        seen.update(self.val[user])
        if mode == 'test':
            seen.update(self.test[user])
        while len(item_idx) < self.sample_size:
            sampled_ids = np.random.choice(list(range(self.itemnum)), 2 * self.sample_size, replace=False)
            sampled_ids = [x for x in sampled_ids if x not in seen and x not in item_idx]
            item_idx.extend(sampled_ids[:])
        return item_idx[:self.sample_size]
