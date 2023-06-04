import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSuperModule(nn.Module):
    def __init__(self, rec_choice, ind_choice):
        super(BaseSuperModule, self).__init__()
        self.rec_choice = rec_choice
        self.ind_choice = ind_choice
        self.rec_size = len(rec_choice)
        self.ind_size = len(ind_choice)

    def _get_position(self, weight, choice):
        i1 = np.where(choice>weight)[0][0]
        i0 = i1 - 1
        p0 = (weight - choice[i0]) / (choice[i1] - choice[i0])
        return i0, i1, p0, 1-p0

    def _get_shared(self, cand):
        shared_idx = []
        weights = []
        shared_decoder_idx = []
        decoder_weights = []
        num_layers = int(len(cand) / 2)
        for i in range(num_layers):
            rec = cand[2 * i]
            ind = cand[2 * i + 1]
            i0, i1, p0, p1 = self._get_position(rec, self.rec_choice)
            i2, i3, p2, p3 = self._get_position(ind, self.ind_choice)
            idx_0 = i0 * self.rec_size + i2
            idx_1 = i1 * self.rec_size + i2
            idx_2 = i0 * self.rec_size + i3
            idx_3 = i1 * self.rec_size + i3
            shared_idx.append((idx_0, idx_1, idx_2, idx_3))
            weights.append((p1 * p3, p0 * p3, p1 * p2, p0 * p2))
        shared_decoder_idx.reverse()
        decoder_weights.reverse()
        return shared_idx, weights, shared_decoder_idx, decoder_weights

    def set_choice(self, cand):
        '''
        cand has shape max_block * 2, self.choice must have shape max_block
        [recon_layer_0, ind_layer_0, recon_layer_1, ind_layer_1, ...]
        when choose a weight of recon_layer_0, ind_layer_0, we find the bound of idx so that
        self.rec_choice[i0] < recon_layer_0 < self.rec_choice[i1]
        self.ind_choice[i2] < ind_layer_0 < self.ind_choice[i3]
        idx_0 = i0 * rec_size + i2
        idx_1 = i1 * rec_size + i2
        idx_2 = i0 * rec_size + i3
        idx_3 = i1 * rec_size + i3
        '''
        self.shared_idx, self.shared_weights, self.shared_decoder_idx, self.shared_decoder_weights = self._get_shared(cand)        # a list of [idx_0, idx_1, idx_2, idx_3], len(shared_idx) = num_layers
        # self.weights = torch.tensor(self.weights)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Base module has no forward method.")
