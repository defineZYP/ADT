import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import *

class BertModel(nn.Module):
    def __init__(self, usernum, itemnum, args):
        super(BertModel, self).__init__()
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = args.maxlen
        self.num_heads = args.num_heads
        self.num_layers = args.num_layers
        self.dev = args.device
        self.dropout = args.dropout
        self.hidden_units = args.hidden_units
        # embedding for bert
        self.item_emb = BertEmbedding(
            vocab_size=itemnum + 100,
            type_vocab_size=args.type_vocab_size,
            maxlen=args.maxlen,
            hidden_units=args.hidden_units,
            dropout=args.dropout
        )
        self.encoder = Encoder(
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_key=int(args.hidden_units/args.num_heads),
            d_value=int(args.hidden_units/args.num_heads),
            hidden_units=args.hidden_units,
            inner_units=args.inner_units,
            attention_dropout=args.attention_dropout,
            act='gelu'
        )
        self.decoder = Decoder(
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_key=int(args.hidden_units/args.num_heads),
            d_value=int(args.hidden_units/args.num_heads),
            hidden_units=args.hidden_units,
            inner_units=args.inner_units,
            attention_dropout=args.attention_dropout,
            act='gelu'
        )
        self.mask_trans_feat = nn.Linear(
            self.hidden_units,
            self.hidden_units
        )
        self.act = nn.GELU()
        # get bias
        mask_bias = np.zeros(itemnum + 100, dtype=np.float32)
        self.mask_bias = nn.Parameter(torch.from_numpy(mask_bias))
        self.mask_layer_norm = nn.LayerNorm(args.hidden_units, eps=1e-5)
        # now didn't try decoder
        # self.decoders = nn.ModuleList()
        self.dev = args.device

    def log2feats(self, src_ids, seq_pos_ids, seq_sent_ids):
        seqs = self.item_emb(src_ids.to(self.dev), seq_pos_ids, seq_sent_ids)
        mask = (src_ids > 0).unsqueeze(1).repeat(1, src_ids.size(1), 1).unsqueeze(1).to(self.dev)
        logits, enc_inputs, ind_outputs = self.encoder(
            seqs,
            mask
        )
        return logits, enc_inputs, ind_outputs, mask
    
    def decode(self, dec_ids, dec_pos_ids, dec_sent_ids, src_logits, src_mask):
        dec_seqs = self.item_emb(dec_ids.to(self.dev), dec_pos_ids, dec_sent_ids)
        mask = (dec_ids > 0).unsqueeze(1).repeat(1, dec_ids.size(1), 1).unsqueeze(1).to(self.dev)
        _, dec_outputs = self.decoder(
            dec_seqs,
            src_logits,
            mask,
            src_mask
        )
        return dec_outputs
    
    def downstream(self, logits):
        mask_logits = self.mask_trans_feat(
            logits
        )
        mask_logits = self.act(mask_logits)
        mask_logits = self.mask_layer_norm(mask_logits)
        token_emb_param = self.item_emb.get_item_emb()
        # TODO
        logits = torch.matmul(mask_logits, token_emb_param.transpose(-2, -1))
        logits += self.mask_bias
        return logits

    def forward(self, src_ids, dec_ids, seq_pos_ids, seq_sent_ids, deq_pos_ids, deq_sent_ids):
        logits, enc_inputs, ind_outputs, src_mask = self.log2feats(
            src_ids,
            seq_pos_ids,
            seq_sent_ids
        )
        dec_outputs = self.decode(
            dec_ids,
            deq_pos_ids,
            deq_sent_ids,
            logits,
            src_mask
        )
        logits = self.downstream(
            logits
        )
        return logits, enc_inputs, dec_outputs, ind_outputs

    def predict(self, user_ids, seqs, seq_pos_ids, seq_sent_ids, candidates):
        log_feats, _, _, _ = self.log2feats(seqs, seq_pos_ids, seq_sent_ids)
        log_feats = self.downstream(log_feats)
        # log_feats = self.out_linear(log_feats)
        log_feats = log_feats[:, -1, :]
        log_feats = log_feats.gather(1, candidates.to(log_feats.device)) # 按列排序找出来结果
        return log_feats
        