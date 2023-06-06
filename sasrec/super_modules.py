import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import *
from base_super_modules import BaseSuperModule

class SuperEncoder(BaseSuperModule):
    def __init__(self, num_layers, hidden_units, num_heads, dropout, rec_choice, ind_choice):
        super(SuperEncoder, self).__init__(rec_choice, ind_choice)
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout = dropout

        self.choice_block_size = self.rec_size * self.ind_size

        self.encoder_layers = nn.ModuleList(
            nn.ModuleList(
                EncoderLayer(
                    hidden_units,
                    num_heads,
                    dropout
                ) for _ in range(self.choice_block_size)
            ) for _ in range(num_layers)
        )

        self.shared_idx = [[0, 0, 0, 0] for _ in range(num_layers)]
        self.shared_weights = [[0, 0, 0 ,0] for _ in range(num_layers)]
        
    def forward(self, seqs, timeline_mask, attention_mask):
        enc_inputs, ind_outputs = [], []
        for layer, idxs, weights in zip(self.encoder_layers, self.shared_idx, self.shared_weights):
            seqs_list = []
            ind_outputs_list = []
            enc_inputs.append(seqs)
            for idx, weight in zip(idxs, weights):
                c_seqs, c_ind_outputs, _ = layer[idx](seqs, timeline_mask, attention_mask)
                seqs_list.append(c_seqs * weight)
                ind_outputs_list.append(c_ind_outputs * weight)
            seqs_list = torch.stack(seqs_list)
            ind_outputs_list = torch.stack(ind_outputs_list)
            seqs = torch.sum(seqs_list, dim=0)
            ind_output = torch.sum(ind_outputs_list, dim=0)
            ind_outputs.append(F.log_softmax(ind_output, dim=3))
        return seqs, enc_inputs, ind_outputs

class SuperDecoder(BaseSuperModule):
    def __init__(self, num_layers, hidden_units, num_heads, dropout, rec_choice, ind_choice):
        super(SuperDecoder, self).__init__(rec_choice, ind_choice)
        self.num_layers = num_layers
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.dropout = dropout

        self.choice_block_size = self.rec_size * self.ind_size

        self.decoder_layers = nn.ModuleList(
            nn.ModuleList(
                DecoderLayer(
                    hidden_units,
                    num_heads,
                    dropout
                ) for _ in range(self.choice_block_size)
            ) for _ in range(num_layers)
        )
        self.shared_idx = [[0, 0, 0, 0] for _ in range(num_layers)]
        self.shared_weights = [[0, 0, 0, 0] for _ in range(num_layers)]

    def forward(self, seqs, encode_outputs, timeline_mask, attention_mask, src_masks):
        dec_outputs = []
        for layer, idxs, weights in zip(self.decoder_layers, self.shared_idx, self.shared_weights):
            seqs_list = []
            for idx, weight in zip(idxs, weights):
                c_seqs, _, _ = layer[idx](seqs, encode_outputs, timeline_mask=timeline_mask, slf_attn_mask=attention_mask, enc_attn_mask=src_masks)
                seqs_list.append(c_seqs * weight)
            seqs_list = torch.stack(seqs_list)
            seqs = torch.sum(seqs_list, dim=0)
            dec_outputs.append(seqs)
        dec_outputs.reverse()
        return seqs, dec_outputs
        