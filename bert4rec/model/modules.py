import os
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_super_modules import BaseSuperModule

class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, type_vocab_size, maxlen, hidden_units, dropout):
        super(BertEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        # print(vocab_size)
        # word_emb
        self.word_emb = nn.Embedding(
            vocab_size,
            hidden_units,
            padding_idx=0
        )
        self.pos_emb = nn.Embedding(
            maxlen,
            hidden_units,
            padding_idx=0
        )
        self.sent_emb = nn.Embedding(
            type_vocab_size,
            hidden_units,
            padding_idx=0
        )
        self.layer_norm = nn.LayerNorm(
            hidden_units,
            eps=1e-5
        )
        self.dropout = nn.Dropout(p=dropout)
    
    def get_item_emb(self):
        return self.word_emb.get_parameter('weight')

    def forward(self, x, pos_ids, sent_ids):
        seqs = self.word_emb(x)
        seqs = seqs + self.pos_emb(pos_ids)
        seqs = seqs + self.sent_emb(sent_ids)
        seqs = self.layer_norm(seqs)
        seqs = self.dropout(seqs)
        return seqs

class MultiHeadAttention(nn.Module):
    def __init__(self, d_key, d_value, hidden_units, num_heads, attention_dropout):
        super(MultiHeadAttention, self).__init__()
        self.d_key = d_key
        self.d_value = d_value
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.query_transfer = nn.Linear(
            hidden_units,
            d_key * num_heads
        )
        self.key_transfer = nn.Linear(
            hidden_units,
            d_key * num_heads
        )
        self.value_transfer = nn.Linear(
            hidden_units,
            d_value * num_heads
        )
        self.out_transfer = nn.Linear(
            d_key * num_heads,
            hidden_units
        )

    def forward(self, queries, keys, values, mask):
        # have difference
        q = self.query_transfer(queries)
        k = self.key_transfer(keys)
        v = self.value_transfer(values)
        hidden_size = q.shape[-1]
        batch_size = queries.shape[0]
        q = q.view(batch_size, -1, self.num_heads, self.d_key)
        q = q.transpose(1, 2) # [batch_size, n_head, max_len, hidden_size]
        k = k.view(batch_size, -1, self.num_heads, self.d_key)
        k = k.transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_value)
        v = v.transpose(1, 2)
        attention_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_key)
        attention_weights = attention_weights.masked_fill(
            mask == 0, -1e9
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        weights = F.dropout(
            attention_weights,
            p=self.attention_dropout,
            training=self.training
        )
        logits = torch.matmul(weights, v)
        logits = logits.transpose(1, 2)
        logits = logits.reshape(batch_size, -1, self.num_heads * self.d_key)
        output = self.out_transfer(logits)
        return output, logits.view(batch_size, -1, self.num_heads, self.d_key)

class DropResidualNormalizeLayer(nn.Module):
    def __init__(self, hidden_units, attention_dropout):
        super(DropResidualNormalizeLayer, self).__init__()
        self.dropout = nn.Dropout(p=attention_dropout)
        self.layer_norm = nn.LayerNorm(
            hidden_units,
            eps=1e-5
        )

    def forward(self, out, prev_out=None):
        logits = self.dropout(out)
        if prev_out is not None:
            logits = logits + prev_out
        return self.layer_norm(logits)

class FFN(nn.Module):
    def __init__(self, hidden_units, inner_units, act):
        super(FFN, self).__init__()
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'gelu':
            self.act = nn.GELU()
        self.fc1 = nn.Linear(
            hidden_units,
            inner_units
        )
        self.fc2 = nn.Linear(
            inner_units,
            hidden_units
        )

    def forward(self, logits):
        logits = self.fc1(logits)
        logits = self.act(logits)
        logits = self.fc2(logits)
        return logits

class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_key, d_value, hidden_units, inner_units, attention_dropout, act):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            d_key=d_key,
            d_value=d_value,
            hidden_units=hidden_units,
            num_heads=num_heads,
            attention_dropout=attention_dropout
        )
        self.drop_residual_normalize_layer_after_multi = DropResidualNormalizeLayer(
            hidden_units=hidden_units,
            attention_dropout=attention_dropout
        )
        self.ffn = FFN(
            hidden_units=hidden_units,
            inner_units=inner_units,
            act=act
        )
        self.drop_residual_normalize_layer_final = DropResidualNormalizeLayer(
            hidden_units=hidden_units,
            attention_dropout=attention_dropout
        )
        self.head_classifier = nn.Linear(int(hidden_units/num_heads), num_heads)

    def forward(self, logits, mask):
        multi_logits, ind_output = self.multi_head_attention(
            queries=logits,
            keys=logits,
            values=logits,
            mask=mask
        )
        logits = self.drop_residual_normalize_layer_after_multi(
            prev_out=logits,
            out=multi_logits
        )
        ffn_logits = self.ffn(
            logits
        )
        logits = self.drop_residual_normalize_layer_final(
            prev_out=logits,
            out=ffn_logits
        )
        return logits, self.head_classifier(ind_output)

class Encoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_key, d_value, hidden_units, inner_units, attention_dropout, act):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.inner_units = inner_units
        self.attention_dropout = attention_dropout
        self.act = act
        self.encoder_layers = nn.ModuleList(
            EncoderLayer(
                num_heads=num_heads,
                d_key=d_key,
                d_value=d_value,
                hidden_units=hidden_units,
                inner_units=inner_units,
                attention_dropout=attention_dropout,
                act=act
            ) for _ in range(num_layers)
        )

    def forward(self, logits, mask):
        enc_inputs = []
        ind_outputs = []
        for enc in self.encoder_layers:
            enc_inputs.append(logits)
            logits, ind_output = enc(logits, mask)
            ind_output = F.log_softmax(ind_output, dim=3)
            ind_outputs.append(ind_output)
        return logits, enc_inputs, ind_outputs
    
class SuperEncoder(BaseSuperModule):
    def __init__(self, num_layers, num_heads, d_key, d_value, hidden_units, inner_units, attention_dropout, act, rec_choice, ind_choice):
        super(SuperEncoder, self).__init__(rec_choice, ind_choice)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.inner_units = inner_units
        self.attention_dropout = attention_dropout
        self.act = act
        self.choice_block_size = self.rec_size * self.ind_size
        self.encoder_layers = nn.ModuleList(
            nn.ModuleList(
                EncoderLayer(
                    num_heads=num_heads,
                    d_key=d_key,
                    d_value=d_value,
                    hidden_units=hidden_units,
                    inner_units=inner_units,
                    attention_dropout=attention_dropout,
                    act=act
                ) for _ in range(self.choice_block_size)
            ) for _ in range(num_layers)
        )
        self.shared_idx = [[0, 0, 0, 0] for _ in range(num_layers)]
        self.shared_weights = [[0, 0, 0, 0] for _ in range(num_layers)]

    def forward(self, logits, mask):
        enc_inputs = []
        ind_outputs = []
        for layer, idxs, weights in zip(self.encoder_layers, self.shared_idx, self.shared_weights):
            seqs_list = []
            ind_outputs_list = []
            enc_inputs.append(logits)
            for idx, weight in zip(idxs, weights):
                c_logits, c_ind_outputs = layer[idx](logits, mask)
                seqs_list.append(c_logits * weight)
                ind_outputs_list.append(c_ind_outputs * weight)
            seqs_list = torch.stack(seqs_list)
            ind_outputs_list = torch.stack(ind_outputs_list)
            logits = torch.sum(seqs_list, dim=0)
            ind_output = torch.sum(ind_outputs_list, dim=0)
            ind_outputs.append(F.log_softmax(ind_output, dim=3))
        return logits, enc_inputs, ind_outputs

########################## Decoder #################################
class DecoderLayer(nn.Module):
    def __init__(self, num_heads, d_key, d_value, hidden_units, inner_units, attention_dropout, act):
        super(DecoderLayer, self).__init__()
        self.dec_multi_head_attention = MultiHeadAttention(
            d_key=d_key,
            d_value=d_value,
            hidden_units=hidden_units,
            num_heads=num_heads,
            attention_dropout=attention_dropout
        )
        self.drop_residual_normalize_layer_after_multi = DropResidualNormalizeLayer(
            hidden_units=hidden_units,
            attention_dropout=attention_dropout
        )
        self.src_dec_attention = MultiHeadAttention(
            d_key=d_key,
            d_value=d_value,
            hidden_units=hidden_units,
            num_heads=num_heads,
            attention_dropout=attention_dropout
        )
        self.drop_residual_normalize_layer_after_src_dec = DropResidualNormalizeLayer(
            hidden_units=hidden_units,
            attention_dropout=attention_dropout
        )
        self.ffn = FFN(
            hidden_units=hidden_units,
            inner_units=inner_units,
            act=act
        )
        self.drop_residual_normalize_layer_final = DropResidualNormalizeLayer(
            hidden_units=hidden_units,
            attention_dropout=attention_dropout
        )

    def forward(self, logits, src_logits, mask, src_mask):
        multi_logits, _ = self.dec_multi_head_attention(
            queries=logits,
            keys=logits,
            values=logits,
            mask=mask
        )
        logits = self.drop_residual_normalize_layer_after_multi(
            prev_out=logits,
            out=multi_logits
        )
        src_dec_logits, _ = self.src_dec_attention(
            queries=logits,
            keys=src_logits,
            values=src_logits,
            mask=src_mask
        )
        logits = self.drop_residual_normalize_layer_after_src_dec(
            prev_out=logits,
            out=src_dec_logits
        )
        ffn_logits = self.ffn(
            logits
        )
        logits = self.drop_residual_normalize_layer_final(
            prev_out=logits,
            out=ffn_logits
        )
        return logits

class Decoder(nn.Module):
    def __init__(self, num_layers, num_heads, d_key, d_value, hidden_units, inner_units, attention_dropout, act):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.inner_units = inner_units
        self.attention_dropout = attention_dropout
        self.act = act
        self.decoder_layers = nn.ModuleList(
            DecoderLayer(
                num_heads=num_heads,
                d_key=d_key,
                d_value=d_value,
                hidden_units=hidden_units,
                inner_units=inner_units,
                attention_dropout=attention_dropout,
                act=act
            ) for _ in range(num_layers)
        )

    def forward(self, logits, src_logits, mask, src_masks):
        dec_outputs = []
        for enc in self.decoder_layers:
            logits = enc(logits, src_logits, mask, src_masks)
            dec_outputs.append(logits)
        dec_outputs.reverse()
        return logits, dec_outputs

class SuperDecoder(BaseSuperModule):
    def __init__(self, num_layers, num_heads, d_key, d_value, hidden_units, inner_units, attention_dropout, act, rec_choice, ind_choice):
        super(SuperDecoder, self).__init__(rec_choice, ind_choice)
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.inner_units = inner_units
        self.attention_dropout = attention_dropout
        self.act = act
        self.choice_block_size = self.rec_size * self.ind_size
        self.decoder_layers = nn.ModuleList(
            nn.ModuleList(
                DecoderLayer(
                    num_heads=num_heads,
                    d_key=d_key,
                    d_value=d_value,
                    hidden_units=hidden_units,
                    inner_units=inner_units,
                    attention_dropout=attention_dropout,
                    act=act
                ) for _ in range(self.choice_block_size)
            ) for _ in range(num_layers)
        )
        self.shared_idx = [[0, 0, 0, 0] for _ in range(num_layers)]
        self.shared_weights = [[0, 0, 0, 0] for _ in range(num_layers)]

    def forward(self, logits, src_logits, mask, src_masks):
        dec_outputs = []
        for layer, idxs, weights in zip(self.decoder_layers, self.shared_idx, self.shared_weights):
            seqs_list = []
            for idx, weight in zip(idxs, weights):
                c_logits = layer[idx](logits, src_logits, mask, src_masks)
                seqs_list.append(c_logits)
            seqs_list = torch.stack(seqs_list)
            logits = torch.sum(seqs_list, dim=0)
            dec_outputs.append(logits)
        dec_outputs.reverse()
        return logits, dec_outputs