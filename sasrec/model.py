import time
import numpy as np
import torch
from modules import MultiHeadAttention, MultiheadAttentionADT, PointWiseFeedForward, Encoder, Decoder
# from utils import hsic_loss

class SASRecADT(torch.nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SASRecADT, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = args.device
        self.num_heads = args.num_heads
        self.maxlen = args.maxlen
        self.num_layers = args.num_layers

        self.item_emb = torch.nn.Embedding(self.item_num+1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.maxlen, args.hidden_units) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout)

        # encoder
        self.encoder = Encoder(args.num_layers, args.hidden_units, args.num_heads, args.dropout)

        # decoder
        self.decoder = Decoder(args.num_layers, args.hidden_units, args.num_heads, args.dropout)

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        self.args = args

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

        seqs, encoder_layer_input, rec_layer_ind, attn_scores = self.encoder(seqs, timeline_mask, attention_mask)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats, encoder_layer_input, rec_layer_ind, attention_mask, attn_scores[-1]

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
        # print(self.training)
        log_feats, encoder_layer_input, rec_layer_ind, src_masks, _ = self.log2feats(log_seqs) # user_ids hasn't been used yet
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
        log_feats, _, _, _, _ = self.log2feats(log_seqs) # user_ids hasn't been used yet
        e = time.time()
        
        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste # [batch_size, hidden_units]

        if not full:
            item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # [batch_size, candidates, hidden_units]
        else:
            item_embs = self.item_emb.weight

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1) # [batch_size, candidates]
        return logits # preds # (U, candidates)