import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import *
from super_modules import *

class SuperSASRecModel(nn.Module):
    def __init__(self, usernum, itemnum, rec_choice, ind_choice, args):
        super(SuperSASRecModel, self).__init__()
        self.usernum = usernum
        self.itemnum = itemnum
        self.dev = args.device
        self.num_heads = args.num_heads
        self.maxlen = args.maxlen
        self.choice = np.zeros(args.num_layers)
        self.num_layers = args.num_layers

        self.item_emb = torch.nn.Embedding(self.itemnum+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout)

        self.encoder = SuperEncoder(
            args.num_layers, 
            args.hidden_units, 
            args.num_heads, 
            args.dropout, 
            rec_choice, 
            ind_choice
        )

        self.decoder = SuperDecoder(
            args.num_layers, 
            args.hidden_units, 
            args.num_heads, 
            args.dropout, 
            rec_choice, 
            ind_choice
        )
    
    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        seqs, encoder_layer_input, rec_layer_ind = self.encoder(seqs, timeline_mask, attention_mask)

        return seqs, encoder_layer_input, rec_layer_ind, attention_mask

    def decode(self, dec_seqs, encode_outputs, src_masks):
        seqs = self.item_emb(torch.LongTensor(dec_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(dec_seqs.shape[1])), [dec_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        timeline_mask = torch.BoolTensor(dec_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        tl = seqs.shape[1] # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        seqs, decoder_layer_output = self.decoder(seqs, encode_outputs, timeline_mask, attention_mask, src_masks)
        return seqs, decoder_layer_output

    def forward(self, user_ids, log_seqs, dec_seqs, pos_seqs, neg_seqs): # for training
        log_feats, encoder_layer_input, rec_layer_ind, src_masks = self.log2feats(log_seqs) # user_ids hasn't been used yet
        dec_outputs, decoder_layer_output = self.decode(dec_seqs, log_feats.transpose(0,1), src_masks)

        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)

        return pos_logits, neg_logits, encoder_layer_input, decoder_layer_output, rec_layer_ind # pos_pred, neg_pred

    def predict(self, user_ids, log_seqs, item_indices, full=False): # for inference
        # print(self.training)
        s = time.time()
        log_feats, _, _, _ = self.log2feats(log_seqs) # user_ids hasn't been used yet
        e = time.time()
        
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste # [batch_size, hidden_units]

        if not full:
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # [batch_size, candidates, hidden_units]
        else:
            item_embs = self.item_emb.weight

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) # [batch_size, candidates]
        return logits # preds # (U, candidates)
        
    def set_choice(self, cand):
        self.encoder.set_choice(cand)
        self.decoder.set_choice(cand)