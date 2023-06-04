import numpy as np

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import DistLayer, DistDecLayer

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
        num_blocks = int(len(cand) / 2)
        for i in range(num_blocks):
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
        self.shared_idx, self.shared_weights, self.shared_decoder_idx, self.shared_decoder_weights = self._get_shared(cand)        # a list of [idx_0, idx_1, idx_2, idx_3], len(shared_idx) = num_blocks
        # self.weights = torch.tensor(self.weights)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Base module has no forward method.")

class SuperDistSAEncoder(BaseSuperModule):               
    def __init__(self, args, rec_choice, ind_choice):
        super(SuperDistSAEncoder, self).__init__(rec_choice, ind_choice)
        layer = DistLayer(args)
        self.choice_block_size = self.rec_size * self.ind_size
        self.layer = nn.ModuleList(
            nn.ModuleList(
                [copy.deepcopy(layer) for _ in range(self.choice_block_size)]
            ) for _ in range(args.num_layers)
        )
        self.shared_idx = [[0, 0, 0, 0] for _ in range(args.num_layers)]
        self.shared_weights = [[0, 0, 0, 0] for _ in range(args.num_layers)]

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        all_encoder_inputs = []
        all_encoder_recs = []
        for layer_module, idxs, weights in zip(self.layer, self.shared_idx, self.shared_weights):
            all_encoder_inputs.append([mean_hidden_states, cov_hidden_states])
            mean_hidden_states_list = []
            cov_hidden_states_list = []
            rec_mean_list = []
            rec_cov_list = []
            for idx, weight in zip(idxs, weights):
                mean_hidden_states, cov_hidden_states, att_scores, rec_mean, rec_cov = layer_module[idx](mean_hidden_states, cov_hidden_states, attention_mask)
                mean_hidden_states_list.append(mean_hidden_states * weight)
                cov_hidden_states_list.append(cov_hidden_states * weight)
                rec_mean_list.append(rec_mean * weight)
                rec_cov_list.append(rec_cov * weight)
            mean_hidden_states_list = torch.stack(mean_hidden_states_list)
            cov_hidden_states_list = torch.stack(cov_hidden_states_list)
            rec_mean_list = torch.stack(rec_mean_list)
            rec_cov_list = torch.stack(rec_cov_list)
            mean_hidden_states = torch.sum(mean_hidden_states_list, dim=0)
            cov_hidden_states = torch.sum(cov_hidden_states_list, dim=0)
            rec_mean = torch.sum(rec_mean_list, dim=0)
            rec_cov = torch.sum(rec_cov_list, dim=0)
            rec_mean = F.log_softmax(rec_mean, dim=3)
            rec_cov = F.log_softmax(rec_cov, dim=3)
            if output_all_encoded_layers:
                all_encoder_layers.append([mean_hidden_states, cov_hidden_states, att_scores])
            all_encoder_recs.append([rec_mean, rec_cov])
        if not output_all_encoded_layers:
            all_encoder_layers.append([mean_hidden_states, cov_hidden_states, att_scores])
        return all_encoder_layers, all_encoder_inputs, all_encoder_recs

class SuperDistSADecoder(BaseSuperModule):
    def __init__(self, args, rec_choice, ind_choice):
        super(SuperDistSADecoder, self).__init__(rec_choice, ind_choice)
        layer = DistDecLayer(args)
        self.choice_block_size = self.rec_size * self.ind_size
        self.layer = nn.ModuleList(
            nn.ModuleList(
                [copy.deepcopy(layer) for _ in range(self.choice_block_size)]
            ) for _ in range(args.num_layers)
        )
        self.shared_idx = [[0, 0, 0, 0] for _ in range(args.num_layers)]
        self.shared_weights = [[0, 0, 0, 0] for _ in range(args.num_layers)]
    
    def forward(self, mean_hidden_states, cov_hidden_states, enc_hidden_states, enc_cov_states, attention_mask, trg_attention_mask):
        decoder_layers = []
        for layer_module, idxs, weights in zip(self.layer, self.shared_idx, self.shared_weights):
            mean_hidden_states_list = []
            cov_hidden_states_list = []
            for idx, weight in zip(idxs, weights):
                mean_hidden_states, cov_hidden_states = layer_module[idx](mean_hidden_states, cov_hidden_states, enc_hidden_states, enc_cov_states, attention_mask, trg_attention_mask)
                mean_hidden_states_list.append(mean_hidden_states * weight)
                cov_hidden_states_list.append(cov_hidden_states * weight)
            mean_hidden_states_list = torch.stack(mean_hidden_states_list)
            cov_hidden_states_list = torch.stack(cov_hidden_states_list)
            mean_hidden_states = torch.sum(mean_hidden_states_list, dim=0)
            cov_hidden_states = torch.sum(cov_hidden_states_list, dim=0)
            decoder_layers.append([mean_hidden_states, cov_hidden_states])
        return decoder_layers
