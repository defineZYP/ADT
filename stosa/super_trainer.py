import os
import numpy as np
import tqdm
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric, cal_mrr
from modules import wasserstein_distance, kl_distance, wasserstein_distance_matmul, d2s_gaussiannormal, d2s_1overx, kl_distance_matmul
from trainer import Trainer
import time

class SuperDistSAModelTrainer(Trainer):
    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(SuperDistSAModelTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        self.args = args

    def bpr_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids):
        # [batch seq_len hidden_units]
        activation = nn.ELU()
        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)
        pos_cov_emb = activation(self.model.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)
        neg_cov_emb = activation(self.model.item_cov_embeddings(neg_ids)) + 1

        # [batch*seq_len hidden_units]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_units) # [batch*seq_len hidden_units]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_units) # [batch*seq_len hidden_units]

        if self.args.distance_metric == 'wasserstein':
            pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = wasserstein_distance(pos_mean, pos_cov, neg_mean, neg_cov)
        else:
            pos_logits = kl_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = kl_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = kl_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.maxlen).float() # [batch*seq_len]
        loss = torch.sum(-torch.log(torch.sigmoid(neg_logits - pos_logits + 1e-24)) * istarget) / torch.sum(istarget)

        pvn_loss = self.args.pvn_weight * torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(istarget)
        auc = torch.sum(
            ((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc, pvn_loss

    def ce_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids):
        # [batch seq_len hidden_units]
        activation = nn.ELU()
        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)
        pos_cov_emb = activation(self.model.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)
        neg_cov_emb = activation(self.model.item_cov_embeddings(neg_ids)) + 1

        # [batch*seq_len hidden_units]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_units) # [batch*seq_len hidden_units]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_units) # [batch*seq_len hidden_units]


        #pos_logits = d2s_gaussiannormal(wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov), self.args.kernel_param)
        pos_logits = -wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
        #neg_logits = d2s_gaussiannormal(wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov), self.args.kernel_param)
        neg_logits = -wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.maxlen).float() # [batch*seq_len]

        loss = torch.sum(
            - torch.log(torch.sigmoid(neg_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(pos_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)


        return loss, auc

    def margin_optimization(self, seq_mean_out, seq_cov_out, pos_ids, neg_ids, margins):
        # [batch seq_len hidden_units]
        activation = nn.ELU()
        pos_mean_emb = self.model.item_mean_embeddings(pos_ids)
        pos_cov_emb = activation(self.model.item_cov_embeddings(pos_ids)) + 1
        neg_mean_emb = self.model.item_mean_embeddings(neg_ids)
        neg_cov_emb = activation(self.model.item_cov_embeddings(neg_ids)) + 1

        # [batch*seq_len hidden_units]
        pos_mean = pos_mean_emb.view(-1, pos_mean_emb.size(2))
        pos_cov = pos_cov_emb.view(-1, pos_cov_emb.size(2))
        neg_mean = neg_mean_emb.view(-1, neg_mean_emb.size(2))
        neg_cov = neg_cov_emb.view(-1, neg_cov_emb.size(2))
        seq_mean_emb = seq_mean_out.view(-1, self.args.hidden_units) # [batch*seq_len hidden_units]
        seq_cov_emb = seq_cov_out.view(-1, self.args.hidden_units) # [batch*seq_len hidden_units]

        if self.args.distance_metric == 'wasserstein':
            pos_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = wasserstein_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = wasserstein_distance(pos_mean, pos_cov, neg_mean, neg_cov)
        else:
            pos_logits = kl_distance(seq_mean_emb, seq_cov_emb, pos_mean, pos_cov)
            neg_logits = kl_distance(seq_mean_emb, seq_cov_emb, neg_mean, neg_cov)
            pos_vs_neg = kl_distance(pos_mean, pos_cov, neg_mean, neg_cov)

        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.maxlen).float() # [batch*seq_len]
        loss = torch.sum(torch.clamp(pos_logits - neg_logits, min=0) * istarget) / torch.sum(istarget)
        pvn_loss = self.args.pvn_weight * torch.sum(torch.clamp(pos_logits - pos_vs_neg, 0) * istarget) / torch.sum(istarget)
        auc = torch.sum(
            ((torch.sign(neg_logits - pos_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc, pvn_loss

    
    def dist_predict_full(self, seq_mean_out, seq_cov_out):
        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.model.item_mean_embeddings.weight
        test_item_cov_emb = elu_activation(self.model.item_cov_embeddings.weight) + 1
        #num_items, emb_size = test_item_cov_emb.shape

        #seq_mean_out = seq_mean_out.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, emb_size)
        #seq_cov_out = seq_cov_out.unsqueeze(1).expand(-1, num_items, -1).reshape(-1, emb_size)

        #if args.distance_metric == 'wasserstein':
        #    return wasserstein_distance(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)
        #else:
        #    return kl_distance(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)
        #return d2s_1overx(wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb))
        return wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb)
        #return d2s_gaussiannormal(wasserstein_distance_matmul(seq_mean_out, seq_cov_out, test_item_mean_emb, test_item_cov_emb))

    def kl_predict_full(self, seq_mean_out, seq_cov_out):
        elu_activation = torch.nn.ELU()
        test_item_mean_emb = self.model.item_mean_embeddings.weight
        test_item_cov_emb = elu_activation(self.model.item_cov_embeddings.weight) + 1

        num_items = test_item_mean_emb.shape[0]
        eval_batch_size = seq_mean_out.shape[0]
        moded_num_items = eval_batch_size - num_items % eval_batch_size
        fake_mean_emb = torch.zeros(moded_num_items, test_item_mean_emb.shape[1], dtype=torch.float32).to(self.device)
        fake_cov_emb = torch.ones(moded_num_items, test_item_mean_emb.shape[1], dtype=torch.float32).to(self.device)

        concated_mean_emb = torch.cat((test_item_mean_emb, fake_mean_emb), 0)
        concated_cov_emb = torch.cat((test_item_cov_emb, fake_cov_emb), 0)

        assert concated_mean_emb.shape[0] == test_item_mean_emb.shape[0] + moded_num_items

        num_batches = int(num_items / eval_batch_size)
        if moded_num_items > 0:
            num_batches += 1

        results = torch.zeros(seq_mean_out.shape[0], concated_mean_emb.shape[0], dtype=torch.float32)
        start_i = 0
        for i_batch in range(num_batches):
            end_i = start_i + eval_batch_size

            results[:, start_i:end_i] = kl_distance_matmul(seq_mean_out, seq_cov_out, concated_mean_emb[start_i:end_i, :], concated_cov_emb[start_i:end_i, :])
            #results[:, start_i:end_i] = d2s_gaussiannormal(kl_distance_matmul(seq_mean_out, seq_cov_out, concated_mean_emb[start_i:end_i, :], concated_cov_emb[start_i:end_i, :]))
            start_i += eval_batch_size

        #print(results[:, :5])
        return results[:, :num_items]

    def set_choice(self, cand):
        self.model.set_choice(cand)

    def iteration(self, epoch, dataloader, lambda1=None, lambda2=None, full_sort=False, train=True):

        str_code = "train" if train else "test"

        #rec_data_iter = tqdm.tqdm(enumerate(dataloader),
        #                          desc=f"Recommendation EP_{str_code}:{epoch}",
        #                          total=len(dataloader),
        #                          bar_format="{l_bar}{r_bar}")
        rec_data_iter = dataloader

        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_pvn_loss = 0.0
            rec_avg_auc = 0.0

            #for i, batch in rec_data_iter:
            for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, dec_ids, target_pos, target_neg, _ = batch
                # bpr optimization
                sequence_mean_output, sequence_cov_output, att_scores, margins, enc_inputs, encoder_recs, dec_outputs  = self.model.finetune(input_ids, dec_ids, user_ids)
                #print(att_scores[0, 0, :, :])
                loss, batch_auc, pvn_loss = self.bpr_optimization(sequence_mean_output, sequence_cov_output, target_pos, target_neg)
                #loss, batch_auc, pvn_loss = self.margin_optimization(sequence_mean_output, sequence_cov_output, target_pos, target_neg, margins)
                #loss, batch_auc = self.ce_optimization(sequence_mean_output, sequence_cov_output, target_pos, target_neg)
                # print(f"Loss before: {loss} {pvn_loss}")
                dec_outputs.reverse()
                # reconstruction loss
                for l in range(self.args.num_layers):
                    loss += lambda1[l] * F.mse_loss(enc_inputs[l][0], dec_outputs[l][0])
                    loss += lambda1[l] * F.mse_loss(enc_inputs[l][1], dec_outputs[l][1])
                # print(f"Loss reconstruction: {loss}")
                # batch
                p_batch_size = encoder_recs[0][0].shape[0]
                label = torch.arange(self.args.num_heads)
                label = torch.tile(label, [p_batch_size * self.args.maxlen, 1]).to(encoder_recs[0][0].device)
                for l in range(self.args.num_layers):
                    # print(f"??? {F.nll_loss(encoder_recs[l][0].view(p_batch_size * self.args.maxlen, self.args.num_heads, self.args.num_heads), label)}")
                    loss += lambda2[l] * F.nll_loss(encoder_recs[l][0].view(p_batch_size * self.args.maxlen, self.args.num_heads, self.args.num_heads), label)
                    loss += lambda2[l] * F.nll_loss(encoder_recs[l][1].view(p_batch_size * self.args.maxlen, self.args.num_heads, self.args.num_heads), label)
                # print(f"Loss independence: {loss}")
                loss += pvn_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()
                rec_avg_pvn_loss += pvn_loss.item()
        else:
            # start = time.time()
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                with torch.no_grad():
                    #for i, batch in rec_data_iter:
                    i = 0
                    for batch in rec_data_iter:
                        # 0. batch_data will be sent into the device(GPU or cpu)
                        batch = tuple(t.to(self.device) for t in batch)
                        user_ids, input_ids, dec_ids, target_pos, target_neg, answers = batch
                        recommend_mean_output, recommend_cov_output, _, _, _, _, _ = self.model.finetune(input_ids, dec_ids, user_ids)

                        recommend_mean_output = recommend_mean_output[:, -1, :]
                        recommend_cov_output = recommend_cov_output[:, -1, :]

                        if self.args.distance_metric == 'kl':
                            rating_pred = self.kl_predict_full(recommend_mean_output, recommend_cov_output)
                        else:
                            rating_pred = self.dist_predict_full(recommend_mean_output, recommend_cov_output)

                        rating_pred = rating_pred.cpu().data.numpy().copy()
                        batch_user_index = user_ids.cpu().numpy()
                        rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 1e+24
                        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                        ind = np.argpartition(rating_pred, 40)[:, :40]
                        #ind = np.argpartition(rating_pred, -40)[:, -40:]
                        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                        # ascending order
                        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::]
                        #arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                        if i == 0:
                            pred_list = batch_pred_list
                            answer_list = answers.cpu().data.numpy()
                        else:
                            pred_list = np.append(pred_list, batch_pred_list, axis=0)
                            answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                        i += 1
                    # hit@1 ndcg@1 hit@5 ndcg@5 hit@10 ndcg@10 hit@15 ndcg@15 hit@20 ndcg@20 hit@40 ndcg@40 mrr
                    # end = time.time()
                    # print(f"here: {end - start}")
                    return self.get_full_sort_score(epoch, answer_list, pred_list, _print=False)

    def train(self, epoch, lambda1, lambda2):
        self.iteration(epoch, self.train_dataloader, lambda1=lambda1, lambda2=lambda2)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, None, None, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, None, None, full_sort, train=False)

    def save_checkpoint(self, filename):
        os.makedirs('./checkpoint', exist_ok=True)
        torch.save(self.model.state_dict(), './checkpoint/super.pth')
