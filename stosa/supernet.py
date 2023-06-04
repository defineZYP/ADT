import torch
import torch.nn as nn
from modules import Encoder, LayerNorm
from super_modules import SuperDistSAEncoder, SuperDistSADecoder

# 魔改
class DisenDistSASupernet(nn.Module):
    def __init__(self, args, rec_choice, ind_choice):
        super(DisenDistSASupernet, self).__init__()
        self.item_mean_embeddings = nn.Embedding(args.item_size, args.hidden_units, padding_idx=0)
        self.item_cov_embeddings = nn.Embedding(args.item_size, args.hidden_units, padding_idx=0)
        self.position_mean_embeddings = nn.Embedding(args.maxlen, args.hidden_units)
        self.position_cov_embeddings = nn.Embedding(args.maxlen, args.hidden_units)
        self.user_margins = nn.Embedding(args.num_users, 1)
        self.item_encoder = SuperDistSAEncoder(args, rec_choice, ind_choice)
        self.item_decoder = SuperDistSADecoder(args, rec_choice, ind_choice)
        self.LayerNorm = LayerNorm(args.hidden_units, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.args = args

        self.apply(self.init_weights)

    def add_position_mean_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_mean_embeddings(sequence)
        position_embeddings = self.position_mean_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(sequence_emb)

        return sequence_emb

    def add_position_cov_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_cov_embeddings(sequence)
        position_embeddings = self.position_cov_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        elu_act = torch.nn.ELU()
        sequence_emb = elu_act(self.dropout(sequence_emb)) + 1

        return sequence_emb

    def finetune(self, input_ids, dec_ids, user_ids):

        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()

        dec_mask = (dec_ids > 0).long()
        extended_dec_mask = dec_mask.unsqueeze(1).unsqueeze(2)

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * (-2 ** 32 + 1)

        extended_dec_mask = extended_dec_mask * subsequent_mask
        extended_dec_mask = extended_dec_mask.to(dtype=next(self.parameters()).dtype)
        extended_dec_mask = (1.0 - extended_dec_mask) * (-2 ** 32 + 1)

        mean_sequence_emb = self.add_position_mean_embedding(input_ids)
        cov_sequence_emb = self.add_position_cov_embedding(input_ids)

        dec_mean_emb = self.add_position_mean_embedding(dec_ids)
        dec_cov_emb = self.add_position_cov_embedding(dec_ids)

        item_encoded_layers, item_encoded_inputs, item_encoded_recs = self.item_encoder(mean_sequence_emb,
                                                cov_sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        mean_sequence_output, cov_sequence_output, att_scores = item_encoded_layers[-1]

        # decode
        item_decoded_outputs = self.item_decoder(
            dec_mean_emb,
            dec_cov_emb,
            mean_sequence_output,
            cov_sequence_output,
            extended_dec_mask,
            extended_attention_mask
        )

        margins = self.user_margins(user_ids)
        return mean_sequence_output, cov_sequence_output, att_scores, margins, item_encoded_inputs, item_encoded_recs, item_decoded_outputs

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.01, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_choice(self, cand):
        self.item_encoder.set_choice(cand)
        self.item_decoder.set_choice(cand)