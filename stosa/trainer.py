# -*- coding: utf-8 -*-

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

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        # print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]), flush=True)
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        # print(post_fix, flush=True)
        # with open(self.args.log_file, 'a') as f:
        #     f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix), None

    def get_full_sort_score(self, epoch, answers, pred_list, _print=True):
        recall, ndcg, mrr = [], [], 0
        recall_dict_list = []
        ndcg_dict_list = []
        for k in [1, 5, 10, 15, 20, 40]:
            recall_result, recall_dict_k = recall_at_k(answers, pred_list, k)
            recall.append(recall_result)
            recall_dict_list.append(recall_dict_k)
            ndcg_result, ndcg_dict_k = ndcg_k(answers, pred_list, k)
            ndcg.append(ndcg_result)
            ndcg_dict_list.append(ndcg_dict_k)
        mrr, mrr_dict = cal_mrr(answers, pred_list)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.8f}'.format(recall[0]), "NDCG@1": '{:.8f}'.format(ndcg[0]),
            "HIT@5": '{:.8f}'.format(recall[1]), "NDCG@5": '{:.8f}'.format(ndcg[1]),
            "HIT@10": '{:.8f}'.format(recall[2]), "NDCG@10": '{:.8f}'.format(ndcg[2]),
            "HIT@15": '{:.8f}'.format(recall[3]), "NDCG@15": '{:.8f}'.format(ndcg[3]),
            "HIT@20": '{:.8f}'.format(recall[4]), "NDCG@20": '{:.8f}'.format(ndcg[4]),
            "HIT@40": '{:.8f}'.format(recall[5]), "NDCG@40": '{:.8f}'.format(ndcg[5]),
            "MRR": '{:.8f}'.format(mrr)
        }
        # if _print:
        #     print(post_fix, flush=True)
        #     with open(self.args.log_file, 'a') as f:
        #         f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[2], ndcg[2], recall[3], ndcg[3], recall[4], ndcg[4], recall[5], ndcg[5], mrr], str(post_fix), [recall_dict_list, ndcg_dict_list, mrr_dict]

    def get_pos_items_ranks(self, batch_pred_lists, answers):
        num_users = len(batch_pred_lists)
        batch_pos_ranks = defaultdict(list)
        for i in range(num_users):
            pred_list = batch_pred_lists[i]
            true_set = set(answers[i])
            for ind, pred_item in enumerate(pred_list):
                if pred_item in true_set:
                    batch_pos_ranks[pred_item].append(ind+1)
        return batch_pos_ranks

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location='cuda:0'))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_units]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)
        # [batch*seq_len hidden_units]
        pos = pos_emb.view(-1, pos_emb.size(2))
        neg = neg_emb.view(-1, neg_emb.size(2))
        seq_emb = seq_out.view(-1, self.args.hidden_units) # [batch*seq_len hidden_units]
        pos_logits = torch.sum(pos * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.maxlen).float() # [batch*seq_len]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        auc = torch.sum(
            ((torch.sign(pos_logits - neg_logits) + 1) / 2) * istarget
        ) / torch.sum(istarget)

        return loss, auc
    

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_units]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_units]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):
        # [item_num hidden_units]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_units ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class PretrainTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(PretrainTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def pretrain(self, epoch, pretrain_dataloader):

        desc = f'AAP-{self.args.aap_weight}-' \
               f'MIP-{self.args.mip_weight}-' \
               f'MAP-{self.args.map_weight}-' \
               f'SP-{self.args.sp_weight}'

        pretrain_data_iter = tqdm.tqdm(enumerate(pretrain_dataloader),
                                       desc=f"{self.args.model_name}-{self.args.data_name} Epoch:{epoch}",
                                       total=len(pretrain_dataloader),
                                       bar_format="{l_bar}{r_bar}")

        self.model.train()
        aap_loss_avg = 0.0
        mip_loss_avg = 0.0
        map_loss_avg = 0.0
        sp_loss_avg = 0.0

        for i, batch in pretrain_data_iter:
            # 0. batch_data will be sent into the device(GPU or CPU)
            batch = tuple(t.to(self.device) for t in batch)
            attributes, masked_item_sequence, pos_items, neg_items, \
            masked_segment_sequence, pos_segment, neg_segment = batch

            aap_loss, mip_loss, map_loss, sp_loss = self.model.pretrain(attributes,
                                            masked_item_sequence, pos_items, neg_items,
                                            masked_segment_sequence, pos_segment, neg_segment)

            joint_loss = self.args.aap_weight * aap_loss + \
                         self.args.mip_weight * mip_loss + \
                         self.args.map_weight * map_loss + \
                         self.args.sp_weight * sp_loss

            self.optim.zero_grad()
            joint_loss.backward()
            self.optim.step()

            aap_loss_avg += aap_loss.item()
            mip_loss_avg += mip_loss.item()
            map_loss_avg += map_loss.item()
            sp_loss_avg += sp_loss.item()

        num = len(pretrain_data_iter) * self.args.pre_batch_size
        post_fix = {
            "epoch": epoch,
            "aap_loss_avg": '{:.4f}'.format(aap_loss_avg /num),
            "mip_loss_avg": '{:.4f}'.format(mip_loss_avg /num),
            "map_loss_avg": '{:.4f}'.format(map_loss_avg / num),
            "sp_loss_avg": '{:.4f}'.format(sp_loss_avg / num),
        }
        # print(desc)
        # print(str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(desc) + '\n')
            f.write(str(post_fix) + '\n')

class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
    


    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        #rec_data_iter = tqdm.tqdm(enumerate(dataloader),
        #                          desc="Recommendation EP_%s:%d" % (str_code, epoch),
        #                          total=len(dataloader),
        #                          bar_format="{l_bar}{r_bar}")
        rec_data_iter = dataloader
        if train:
            self.model.train()
            rec_avg_loss = 0.0
            rec_cur_loss = 0.0
            rec_avg_auc = 0.0

            #for i, batch in rec_data_iter:
            for batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, target_pos, target_neg, _ = batch
                # Binary cross_entropy
                sequence_output, _ = self.model.finetune(input_ids)
                loss, batch_auc = self.cross_entropy(sequence_output, target_pos, target_neg)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.4f}'.format(rec_avg_auc / len(rec_data_iter)),
            }

            # if (epoch + 1) % self.args.log_freq == 0:
            #     print(str(post_fix), flush=True)

            # with open(self.args.log_file, 'a') as f:
            #     f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                #for i, batch in rec_data_iter:
                i = 0
                for batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output, _ = self.model.finetune(input_ids)

                    recommend_output = recommend_output[:, -1, :]

                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    ind = np.argpartition(rating_pred, -40)[:, -40:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                    i += 1
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                #for i, batch in rec_data_iter:
                i = 0
                for batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                    i += 1

                return self.get_sample_scores(epoch, pred_list)



class DistSAModelTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args, lambda1, lambda2):
        super(DistSAModelTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        self.args = args
        self.lambda1 = lambda1
        self.lambda2 = lambda2

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


    def iteration(self, epoch, dataloader, full_sort=False, train=True):

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
                    loss += self.lambda1[l] * F.mse_loss(enc_inputs[l][0], dec_outputs[l][0])
                    loss += self.lambda1[l] * F.mse_loss(enc_inputs[l][1], dec_outputs[l][1])
                # print(f"Loss reconstruction: {loss}")
                # independence loss
                p_batch_size = encoder_recs[0][0].shape[0]
                label = torch.arange(self.args.num_heads)
                label = torch.tile(label, [p_batch_size * self.args.maxlen, 1]).to(encoder_recs[0][0].device)
                for l in range(self.args.num_layers):
                    # print(f"??? {F.nll_loss(encoder_recs[l][0].view(p_batch_size * self.args.maxlen, self.args.num_heads, self.args.num_heads), label)}")
                    loss += self.lambda2[l] * F.nll_loss(encoder_recs[l][0].view(p_batch_size * self.args.maxlen, self.args.num_heads, self.args.num_heads), label)
                    loss += self.lambda2[l] * F.nll_loss(encoder_recs[l][1].view(p_batch_size * self.args.maxlen, self.args.num_heads, self.args.num_heads), label)
                # print(f"Loss independence: {loss}")
                loss += pvn_loss
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                rec_avg_loss += loss.item()
                rec_cur_loss = loss.item()
                rec_avg_auc += batch_auc.item()
                rec_avg_pvn_loss += pvn_loss.item()

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": '{:.4f}'.format(rec_avg_loss / len(rec_data_iter)),
                "rec_cur_loss": '{:.4f}'.format(rec_cur_loss),
                "rec_avg_auc": '{:.6f}'.format(rec_avg_auc / len(rec_data_iter)),
                "rec_avg_pvn_loss": '{:.6f}'.format(rec_avg_pvn_loss / len(rec_data_iter)),
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix), flush=True)
        else:
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
                        
                        # print(rating_pred.shape)

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
                    return self.get_full_sort_score(epoch, answer_list, pred_list)
